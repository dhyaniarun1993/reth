//! Command that verifies the OP proofs storage against the canonical chain state.

use alloy_primitives::{B256, U256};
use clap::Parser;
use reth_cli::chainspec::ChainSpecParser;
use reth_cli_commands::common::{AccessRights, CliNodeTypes, Environment, EnvironmentArgs};
use reth_node_core::version::version_metadata;
use reth_optimism_chainspec::OpChainSpec;
use reth_optimism_primitives::OpPrimitives;
use reth_optimism_trie::{db::MdbxProofsStorage, OpProofsStorage, OpProofsStore};
use reth_provider::{BlockNumReader, DatabaseProviderFactory, DBProvider};
use reth_trie::hashed_cursor::{HashedCursor, HashedCursorFactory};
use reth_trie::trie_cursor::{TrieCursor, TrieCursorFactory};
use reth_trie_common::Nibbles;
use reth_trie_db::{DatabaseHashedCursorFactory, DatabaseTrieCursorFactory};
use std::{path::PathBuf, sync::Arc};
use tracing::{error, info, warn};

/// Verifies the proofs storage against the canonical chain state.
///
/// This command walks through both the canonical reth database and the proofs storage,
/// comparing account trie nodes, storage trie nodes, and leaf values to identify any
/// discrepancies that could cause state root mismatches.
#[derive(Debug, Parser)]
pub struct VerifyCommand<C: ChainSpecParser> {
    #[command(flatten)]
    env: EnvironmentArgs<C>,

    /// The path to the storage DB for proofs history.
    #[arg(
        long = "proofs-history.storage-path",
        value_name = "PROOFS_HISTORY_STORAGE_PATH",
        required = true
    )]
    pub storage_path: PathBuf,

    /// Enable verbose output showing all comparisons
    #[arg(long, short = 'v')]
    pub verbose: bool,

    /// Maximum number of differences to report before stopping
    #[arg(long = "max-differences", default_value_t = 100)]
    pub max_differences: usize,
}

impl<C: ChainSpecParser<ChainSpec = OpChainSpec>> VerifyCommand<C> {
    /// Execute [`VerifyCommand`].
    pub async fn execute<N: CliNodeTypes<ChainSpec = C::ChainSpec, Primitives = OpPrimitives>>(
        self,
    ) -> eyre::Result<()> {
        info!(target: "reth::cli", "reth {} starting", version_metadata().short_version);
        info!(target: "reth::cli", "Verifying OP proofs storage at: {:?}", self.storage_path);

        // Initialize the environment with read-only access
        let Environment { provider_factory, .. } = self.env.init::<N>(AccessRights::RO)?;

        let storage: OpProofsStorage<Arc<MdbxProofsStorage>> = Arc::new(
            MdbxProofsStorage::new(&self.storage_path)
                .map_err(|e| eyre::eyre!("Failed to create MdbxProofsStorage: {e}"))?,
        )
        .into();

        // Get latest block from proof storage
        let (proof_latest_block, _) = storage
            .get_latest_block_number()?
            .ok_or_else(|| eyre::eyre!("No blocks in proofs storage"))?;

        // Get latest block from canonical provider
        let canonical_latest_block = provider_factory.last_block_number()?;

        // Determine block to verify at
        let block_number = if canonical_latest_block <= proof_latest_block {
            // Canonical is at or behind proof storage - verify at canonical's latest
            // (proof storage has historical data for this block)
            if canonical_latest_block < proof_latest_block {
                info!(
                    target: "reth::cli",
                    canonical_block = canonical_latest_block,
                    proof_block = proof_latest_block,
                    "Proof storage is ahead - verifying at canonical's latest block"
                );
                let (earliest_proof_block, _) = storage
                    .get_earliest_block_number()?
                    .ok_or_else(|| eyre::eyre!("No blocks in proofs storage"))?;

                if canonical_latest_block < earliest_proof_block {
                    return Err(eyre::eyre!(
                        "Canonical latest block {} is earlier than earliest block in proof storage {}. Cannot verify.",
                        canonical_latest_block,
                        earliest_proof_block
                    ));
                }
            }
            canonical_latest_block
        } else {
            // Canonical is ahead of proof storage - cannot verify latest canonical state
            return Err(eyre::eyre!(
                "Proof storage is behind: canonical is at block {} but proof storage is only at block {}. \
                Cannot verify - proof storage needs to catch up.",
                canonical_latest_block,
                proof_latest_block
            ));
        };

        // TODO: support verifying at a specific block number (not just latest)

        info!(
            target: "reth::cli",
            block_number,
            "Starting verification"
        );

        // Create verification report
        let mut report = VerificationReport::new(block_number);

        // Compare account trie
        info!(target: "reth::cli", "Comparing account trie nodes...");
        self.compare_account_trie(
            &provider_factory,
            &storage,
            block_number,
            &mut report,
        )?;

        // Compare account leaves (hashed accounts) and their storage inline
        info!(target: "reth::cli", "Comparing hashed accounts and storage...");
        self.compare_hashed_accounts(
            &provider_factory,
            &storage,
            block_number,
            &mut report,
        )?;

        // Print report
        report.print(self.verbose);

        if report.has_errors() {
            Err(eyre::eyre!("Verification failed with {} total differences", report.total_differences()))
        } else {
            info!(target: "reth::cli", "✓ Verification passed - no differences found");
            Ok(())
        }
    }

    fn compare_account_trie<Provider>(
        &self,
        canonical_provider: &Provider,
        storage: &OpProofsStorage<Arc<MdbxProofsStorage>>,
        block_number: u64,
        report: &mut VerificationReport,
    ) -> eyre::Result<()>
    where
        Provider: DatabaseProviderFactory,
    {
        use reth_trie::trie_cursor::TrieCursor;

        info!(target: "reth::cli", "Comparing account trie nodes...");

        // Get canonical database provider
        let canonical_db = canonical_provider.database_provider_ro()?.disable_long_read_transaction_safety();
        let canonical_tx = canonical_db.into_tx();

        // Create cursor factory and get account trie cursor
        let cursor_factory = DatabaseTrieCursorFactory::new(&canonical_tx);
        let mut canonical_cursor = cursor_factory.account_trie_cursor()?;

        // Get proof storage cursor for account trie at the specific block
        let mut proof_cursor = storage.account_trie_cursor(block_number)?;

        let mut differences = 0;
        let mut nodes_checked = 0;

        // Iterate through both cursors in parallel
        loop {
            let canonical_entry = canonical_cursor.next()?;
            let proof_entry = proof_cursor.next()?;

            match (canonical_entry, proof_entry) {
                (Some((canonical_nibbles, canonical_node)), Some((proof_nibbles, proof_node))) => {
                    nodes_checked += 1;

                    if canonical_nibbles != proof_nibbles {
                        // Path mismatch - tries have different structure
                        if self.verbose {
                            warn!(
                                target: "reth::cli",
                                canonical_path = ?canonical_nibbles,
                                proof_path = ?proof_nibbles,
                                "Trie structure mismatch - different paths"
                            );
                        }
                        report.account_trie_diffs.push(AccountTrieDiff {
                            path: canonical_nibbles.clone(),
                            canonical_value: format!("Path: {:?}, Node: {:?}", canonical_nibbles, canonical_node),
                            proofs_value: format!("Path: {:?}, Node: {:?}", proof_nibbles, proof_node),
                        });
                        differences += 1;
                    } else if canonical_node != proof_node {
                        // Same path, different node value
                        if self.verbose {
                            warn!(
                                target: "reth::cli",
                                path = ?canonical_nibbles,
                                "Trie node value mismatch"
                            );
                        }
                        report.account_trie_diffs.push(AccountTrieDiff {
                            path: canonical_nibbles,
                            canonical_value: format!("{:?}", canonical_node),
                            proofs_value: format!("{:?}", proof_node),
                        });
                        differences += 1;
                    }

                    if differences >= self.max_differences {
                        error!(
                            target: "reth::cli",
                            "Reached maximum difference limit ({}), stopping comparison",
                            self.max_differences
                        );
                        break;
                    }

                    if self.verbose && nodes_checked % 10000 == 0 {
                        info!(
                            target: "reth::cli",
                            nodes_checked,
                            differences,
                            "Account trie comparison progress"
                        );
                    }
                }
                (Some((canonical_nibbles, canonical_node)), None) => {
                    // Canonical has more entries - proof storage is missing nodes
                    nodes_checked += 1;
                    if self.verbose {
                        warn!(
                            target: "reth::cli",
                            path = ?canonical_nibbles,
                            "Trie node missing in proof storage"
                        );
                    }
                    report.account_trie_diffs.push(AccountTrieDiff {
                        path: canonical_nibbles,
                        canonical_value: format!("{:?}", canonical_node),
                        proofs_value: "MISSING".to_string(),
                    });
                    differences += 1;

                    if differences >= self.max_differences {
                        error!(
                            target: "reth::cli",
                            "Reached maximum difference limit ({}), stopping comparison",
                            self.max_differences
                        );
                        break;
                    }
                }
                (None, Some((proof_nibbles, proof_node))) => {
                    // Proof storage has extra nodes
                    if self.verbose {
                        warn!(
                            target: "reth::cli",
                            path = ?proof_nibbles,
                            "Extra trie node in proof storage"
                        );
                    }
                    report.account_trie_diffs.push(AccountTrieDiff {
                        path: proof_nibbles,
                        canonical_value: "MISSING".to_string(),
                        proofs_value: format!("{:?}", proof_node),
                    });
                    differences += 1;

                    if differences >= self.max_differences {
                        error!(
                            target: "reth::cli",
                            "Reached maximum difference limit ({}), stopping comparison",
                            self.max_differences
                        );
                        break;
                    }
                }
                (None, None) => {
                    // Both exhausted - done
                    break;
                }
            }
        }

        // Continue draining canonical if we stopped early due to max differences
        while canonical_cursor.next()?.is_some() {
            nodes_checked += 1;
        }

        info!(
            target: "reth::cli",
            nodes_checked,
            differences,
            "Account trie node comparison complete"
        );

        report.account_trie_compared = true;
        Ok(())
    }

    fn compare_hashed_accounts<Provider>(
        &self,
        canonical_provider: &Provider,
        storage: &OpProofsStorage<Arc<MdbxProofsStorage>>,
        block_number: u64,
        report: &mut VerificationReport,
    ) -> eyre::Result<()>
    where
        Provider: DatabaseProviderFactory,
    {
        info!(target: "reth::cli", "Comparing hashed account values...");

        // Get canonical database provider
        let canonical_db = canonical_provider.database_provider_ro()?.disable_long_read_transaction_safety();
        let canonical_tx = canonical_db.into_tx();

        // Create cursor factory and get hashed account cursor
        let cursor_factory = DatabaseHashedCursorFactory::new(&canonical_tx);
        let mut canonical_cursor = cursor_factory.hashed_account_cursor()?;

        // Get proof storage cursor for hashed accounts at the specific block
        let mut proof_cursor = storage.account_hashed_cursor(block_number)?;

        let mut differences = 0;
        let mut accounts_checked = 0;

        // Iterate through both cursors in parallel
        loop {
            let canonical_entry = canonical_cursor.next()?;
            let proof_entry = proof_cursor.next()?;

            match (canonical_entry, proof_entry) {
                (Some((canonical_address, canonical_account)), Some((proof_address, proof_account))) => {
                    accounts_checked += 1;

                    if canonical_address != proof_address {
                        // Address mismatch - different accounts present
                        if self.verbose {
                            warn!(
                                target: "reth::cli",
                                canonical_address = ?canonical_address,
                                proof_address = ?proof_address,
                                "Account address mismatch"
                            );
                        }
                        report.hashed_account_diffs.push(HashedAccountDiff {
                            address: canonical_address,
                            balance_diff: Some((canonical_account.balance, U256::ZERO)),
                            nonce_diff: Some((canonical_account.nonce, 0)),
                            code_hash_diff: None,
                            storage_root_diff: None,
                        });
                        differences += 1;
                    } else {
                        // Same address, compare account values
                        let mut diff = HashedAccountDiff {
                            address: canonical_address,
                            balance_diff: None,
                            nonce_diff: None,
                            code_hash_diff: None,
                            storage_root_diff: None,
                        };

                        let mut has_diff = false;

                        if canonical_account.balance != proof_account.balance {
                            diff.balance_diff = Some((canonical_account.balance, proof_account.balance));
                            has_diff = true;
                        }

                        if canonical_account.nonce != proof_account.nonce {
                            diff.nonce_diff = Some((canonical_account.nonce, proof_account.nonce));
                            has_diff = true;
                        }

                        if canonical_account.bytecode_hash != proof_account.bytecode_hash {
                            diff.code_hash_diff = Some((
                                canonical_account.bytecode_hash.unwrap_or_default(),
                                proof_account.bytecode_hash.unwrap_or_default(),
                            ));
                            has_diff = true;
                        }

                        if has_diff {
                            if self.verbose {
                                warn!(
                                    target: "reth::cli",
                                    address = ?canonical_address,
                                    "Account value mismatch"
                                );
                            }
                            report.hashed_account_diffs.push(diff);
                            differences += 1;
                        }

                        // Verify storage for this account
                        // (cursors will return None immediately if account has no storage)
                        self.verify_account_storage(
                            canonical_provider,
                            storage,
                            block_number,
                            canonical_address,
                            report,
                        )?;
                    }

                    if differences >= self.max_differences {
                        error!(
                            target: "reth::cli",
                            "Reached maximum difference limit ({}), stopping comparison",
                            self.max_differences
                        );
                        break;
                    }

                    if self.verbose && accounts_checked % 10000 == 0 {
                        info!(
                            target: "reth::cli",
                            accounts_checked,
                            differences,
                            "Hashed account comparison progress"
                        );
                    }
                }
                (Some((canonical_address, canonical_account)), None) => {
                    // Canonical has more accounts - missing in proof storage
                    accounts_checked += 1;
                    if self.verbose {
                        warn!(
                            target: "reth::cli",
                            address = ?canonical_address,
                            "Account missing in proof storage"
                        );
                    }
                    report.hashed_account_diffs.push(HashedAccountDiff {
                        address: canonical_address,
                        balance_diff: Some((canonical_account.balance, U256::ZERO)),
                        nonce_diff: Some((canonical_account.nonce, 0)),
                        code_hash_diff: None,
                        storage_root_diff: None,
                    });
                    differences += 1;

                    if differences >= self.max_differences {
                        error!(
                            target: "reth::cli",
                            "Reached maximum difference limit ({}), stopping comparison",
                            self.max_differences
                        );
                        break;
                    }
                }
                (None, Some((proof_address, proof_account))) => {
                    // Proof storage has extra accounts
                    if self.verbose {
                        warn!(
                            target: "reth::cli",
                            address = ?proof_address,
                            "Extra account in proof storage"
                        );
                    }
                    report.hashed_account_diffs.push(HashedAccountDiff {
                        address: proof_address,
                        balance_diff: Some((U256::ZERO, proof_account.balance)),
                        nonce_diff: Some((0, proof_account.nonce)),
                        code_hash_diff: None,
                        storage_root_diff: None,
                    });
                    differences += 1;

                    if differences >= self.max_differences {
                        error!(
                            target: "reth::cli",
                            "Reached maximum difference limit ({}), stopping comparison",
                            self.max_differences
                        );
                        break;
                    }
                }
                (None, None) => {
                    // Both exhausted - done
                    break;
                }
            }
        }

        info!(
            target: "reth::cli",
            accounts_checked,
            differences,
            "Hashed account comparison complete"
        );

        report.hashed_accounts_compared = true;
        // Storage trie and hashed storage are checked inline for each account
        report.storage_trie_compared = true;
        report.storage_compared = true;
        Ok(())
    }

    /// Verify both storage trie and hashed storage for an account
    fn verify_account_storage<Provider>(
        &self,
        canonical_provider: &Provider,
        storage: &OpProofsStorage<Arc<MdbxProofsStorage>>,
        block_number: u64,
        hashed_address: B256,
        report: &mut VerificationReport,
    ) -> eyre::Result<()>
    where
        Provider: DatabaseProviderFactory,
    {
        self.compare_storage_trie_by_hash(
            canonical_provider,
            storage,
            block_number,
            hashed_address,
            report,
        )?;
        self.compare_storage_by_hash(
            canonical_provider,
            storage,
            block_number,
            hashed_address,
            report,
        )
    }

    fn compare_storage_trie_by_hash<Provider>(
        &self,
        canonical_provider: &Provider,
        storage: &OpProofsStorage<Arc<MdbxProofsStorage>>,
        block_number: u64,
        hashed_address: B256,
        report: &mut VerificationReport,
    ) -> eyre::Result<()>
    where
        Provider: DatabaseProviderFactory,
    {
        if self.verbose {
            info!(target: "reth::cli", "Comparing storage trie nodes for hashed address {:?}...", hashed_address);
        }

        // Get canonical database provider
        let canonical_db = canonical_provider.database_provider_ro()?.disable_long_read_transaction_safety();
        let canonical_tx = canonical_db.into_tx();

        // Create cursor factory and get storage trie cursor
        let cursor_factory = DatabaseTrieCursorFactory::new(&canonical_tx);
        let mut canonical_cursor = cursor_factory.storage_trie_cursor(hashed_address)?;

        // Get proof storage cursor for storage trie at the specific block
        let mut proof_cursor = storage.storage_trie_cursor(hashed_address, block_number)?;

        let mut differences = 0;
        let mut nodes_checked = 0;

        // Position both cursors at the beginning (empty nibbles)
        // The canonical cursor uses next_dup() internally which requires positioning first
        let mut canonical_entry = canonical_cursor.seek(Nibbles::default())?;
        let mut proof_entry = proof_cursor.seek(Nibbles::default())?;

        // Iterate through both cursors in parallel
        loop {
            match (canonical_entry.take(), proof_entry.take()) {
                (Some((canonical_nibbles, canonical_node)), Some((proof_nibbles, proof_node))) => {
                    nodes_checked += 1;

                    if canonical_nibbles != proof_nibbles {
                        // Path mismatch - storage tries have different structure
                        if self.verbose {
                            warn!(
                                target: "reth::cli",
                                hashed_address = ?hashed_address,
                                canonical_path = ?canonical_nibbles,
                                proof_path = ?proof_nibbles,
                                "Storage trie structure mismatch"
                            );
                        }
                        report.storage_trie_diffs.push(StorageTrieDiff {
                            hashed_address,
                            path: canonical_nibbles.clone(),
                            canonical_value: format!("Path: {:?}, Node: {:?}", canonical_nibbles, canonical_node),
                            proofs_value: format!("Path: {:?}, Node: {:?}", proof_nibbles, proof_node),
                        });
                        differences += 1;
                    } else if canonical_node != proof_node {
                        // Same path, different node value
                        if self.verbose {
                            warn!(
                                target: "reth::cli",
                                hashed_address = ?hashed_address,
                                path = ?canonical_nibbles,
                                "Storage trie node value mismatch"
                            );
                        }
                        report.storage_trie_diffs.push(StorageTrieDiff {
                            hashed_address,
                            path: canonical_nibbles,
                            canonical_value: format!("{:?}", canonical_node),
                            proofs_value: format!("{:?}", proof_node),
                        });
                        differences += 1;
                    }

                    if differences >= self.max_differences {
                        error!(
                            target: "reth::cli",
                            "Reached maximum difference limit ({}), stopping comparison",
                            self.max_differences
                        );
                        break;
                    }

                    // Get next entries
                    canonical_entry = canonical_cursor.next()?;
                    proof_entry = proof_cursor.next()?;
                }
                (Some((canonical_nibbles, canonical_node)), None) => {
                    // Canonical has more entries - proof storage is missing nodes
                    nodes_checked += 1;
                    if self.verbose {
                        warn!(
                            target: "reth::cli",
                            hashed_address = ?hashed_address,
                            path = ?canonical_nibbles,
                            "Storage trie node missing in proof storage"
                        );
                    }
                    report.storage_trie_diffs.push(StorageTrieDiff {
                        hashed_address,
                        path: canonical_nibbles,
                        canonical_value: format!("{:?}", canonical_node),
                        proofs_value: "MISSING".to_string(),
                    });
                    differences += 1;

                    if differences >= self.max_differences {
                        error!(
                            target: "reth::cli",
                            "Reached maximum difference limit ({}), stopping comparison",
                            self.max_differences
                        );
                        break;
                    }

                    // Advance only canonical cursor
                    canonical_entry = canonical_cursor.next()?;
                }
                (None, Some((proof_nibbles, proof_node))) => {
                    // Proof storage has extra nodes
                    if self.verbose {
                        warn!(
                            target: "reth::cli",
                            hashed_address = ?hashed_address,
                            path = ?proof_nibbles,
                            "Extra storage trie node in proof storage"
                        );
                    }
                    report.storage_trie_diffs.push(StorageTrieDiff {
                        hashed_address,
                        path: proof_nibbles,
                        canonical_value: "MISSING".to_string(),
                        proofs_value: format!("{:?}", proof_node),
                    });
                    differences += 1;

                    if differences >= self.max_differences {
                        error!(
                            target: "reth::cli",
                            "Reached maximum difference limit ({}), stopping comparison",
                            self.max_differences
                        );
                        break;
                    }

                    // Advance only proof cursor
                    proof_entry = proof_cursor.next()?;
                }
                (None, None) => {
                    // Both exhausted - done
                    break;
                }
            }
        }

        if self.verbose {
            info!(
                target: "reth::cli",
                hashed_address = ?hashed_address,
                nodes_checked,
                differences,
                "Storage trie node comparison complete"
            );
        }

        Ok(())
    }

    fn compare_storage_by_hash<Provider>(
        &self,
        canonical_provider: &Provider,
        storage: &OpProofsStorage<Arc<MdbxProofsStorage>>,
        block_number: u64,
        hashed_address: B256,
        report: &mut VerificationReport,
    ) -> eyre::Result<()>
    where
        Provider: DatabaseProviderFactory,
    {
        if self.verbose {
            info!(target: "reth::cli", "Comparing storage values for hashed address {:?}...", hashed_address);
        }

        // Get canonical database provider
        let canonical_db = canonical_provider.database_provider_ro()?.disable_long_read_transaction_safety();
        let canonical_tx = canonical_db.into_tx();

        // Create cursor factory and get hashed storage cursor
        let cursor_factory = DatabaseHashedCursorFactory::new(&canonical_tx);
        let mut canonical_cursor = cursor_factory.hashed_storage_cursor(hashed_address)?;

        // Get proof storage cursor for hashed storage at the specific block
        let mut proof_cursor = storage.storage_hashed_cursor(hashed_address, block_number)?;

        let mut differences = 0;
        let mut slots_checked = 0;

        // Position both cursors at the beginning
        // The canonical cursor uses dup cursor which requires positioning first
        let mut canonical_entry = canonical_cursor.seek(B256::ZERO)?;
        let mut proof_entry = proof_cursor.seek(B256::ZERO)?;

        // Iterate through both cursors in parallel
        loop {
            match (canonical_entry.take(), proof_entry.take()) {
                (Some((canonical_key, canonical_value)), Some((proof_key, proof_value))) => {
                    slots_checked += 1;

                    if canonical_key != proof_key {
                        // Slot key mismatch - different slots present
                        if self.verbose {
                            warn!(
                                target: "reth::cli",
                                canonical_slot = ?canonical_key,
                                proof_slot = ?proof_key,
                                "Storage slot key mismatch"
                            );
                        }
                        report.storage_diffs.push(StorageDiff {
                            hashed_address,
                            slot: canonical_key,
                            canonical_value,
                            proofs_value: U256::ZERO,
                        });
                        differences += 1;
                    } else if canonical_value != proof_value {
                        // Same slot, different value
                        if self.verbose {
                            warn!(
                                target: "reth::cli",
                                slot = ?canonical_key,
                                "Storage value mismatch"
                            );
                        }
                        report.storage_diffs.push(StorageDiff {
                            hashed_address,
                            slot: canonical_key,
                            canonical_value,
                            proofs_value: proof_value,
                        });
                        differences += 1;
                    }

                    if differences >= self.max_differences {
                        error!(
                            target: "reth::cli",
                            "Reached maximum difference limit ({}), stopping comparison",
                            self.max_differences
                        );
                        break;
                    }

                    // Get next entries
                    canonical_entry = canonical_cursor.next()?;
                    proof_entry = proof_cursor.next()?;
                }
                (Some((canonical_key, canonical_value)), None) => {
                    // Canonical has more slots - missing in proof storage
                    slots_checked += 1;
                    if self.verbose {
                        warn!(
                            target: "reth::cli",
                            slot = ?canonical_key,
                            "Storage slot missing in proof storage"
                        );
                    }
                    report.storage_diffs.push(StorageDiff {
                        hashed_address,
                        slot: canonical_key,
                        canonical_value,
                        proofs_value: U256::ZERO,
                    });
                    differences += 1;

                    if differences >= self.max_differences {
                        error!(
                            target: "reth::cli",
                            "Reached maximum difference limit ({}), stopping comparison",
                            self.max_differences
                        );
                        break;
                    }

                    // Advance only canonical cursor
                    canonical_entry = canonical_cursor.next()?;
                }
                (None, Some((proof_key, proof_value))) => {
                    // Proof storage has extra slots
                    if self.verbose {
                        warn!(
                            target: "reth::cli",
                            slot = ?proof_key,
                            "Extra storage slot in proof storage"
                        );
                    }
                    report.storage_diffs.push(StorageDiff {
                        hashed_address,
                        slot: proof_key,
                        canonical_value: U256::ZERO,
                        proofs_value: proof_value,
                    });
                    differences += 1;

                    if differences >= self.max_differences {
                        error!(
                            target: "reth::cli",
                            "Reached maximum difference limit ({}), stopping comparison",
                            self.max_differences
                        );
                        break;
                    }

                    // Advance only proof cursor
                    proof_entry = proof_cursor.next()?;
                }
                (None, None) => {
                    // Both exhausted - done
                    break;
                }
            }
        }

        if self.verbose {
            info!(
                target: "reth::cli",
                hashed_address = ?hashed_address,
                slots_checked,
                differences,
                "Storage value comparison complete"
            );
        }

        Ok(())
    }
}

impl<C: ChainSpecParser> VerifyCommand<C> {
    /// Returns the underlying chain being used to run this command
    pub const fn chain_spec(&self) -> Option<&Arc<C::ChainSpec>> {
        Some(&self.env.chain)
    }
}

/// Verification report tracking all differences found
#[derive(Debug)]
struct VerificationReport {
    block_number: u64,
    account_trie_compared: bool,
    hashed_accounts_compared: bool,
    storage_trie_compared: bool,
    storage_compared: bool,
    account_trie_diffs: Vec<AccountTrieDiff>,
    hashed_account_diffs: Vec<HashedAccountDiff>,
    storage_trie_diffs: Vec<StorageTrieDiff>,
    storage_diffs: Vec<StorageDiff>,
}

impl VerificationReport {
    fn new(block_number: u64) -> Self {
        Self {
            block_number,
            account_trie_compared: false,
            hashed_accounts_compared: false,
            storage_trie_compared: false,
            storage_compared: false,
            account_trie_diffs: Vec::new(),
            hashed_account_diffs: Vec::new(),
            storage_trie_diffs: Vec::new(),
            storage_diffs: Vec::new(),
        }
    }

    fn has_errors(&self) -> bool {
        !self.account_trie_diffs.is_empty() ||
            !self.hashed_account_diffs.is_empty() ||
            !self.storage_trie_diffs.is_empty() ||
            !self.storage_diffs.is_empty()
    }

    fn total_differences(&self) -> usize {
        self.account_trie_diffs.len() +
            self.hashed_account_diffs.len() +
            self.storage_trie_diffs.len() +
            self.storage_diffs.len()
    }

    fn print(&self, verbose: bool) {
        println!("\n╔════════════════════════════════════════════════════════════════╗");
        println!("║          Proofs Storage Verification Report                   ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!("\nBlock Number: {}", self.block_number);
        println!("\n─────────────────────────────────────────────────────────────────");

        if !self.has_errors() {
            println!("\n✓ SUCCESS: No differences found!");
            println!("\nChecks performed:");
            if self.account_trie_compared {
                println!("  ✓ Account trie nodes");
            }
            if self.hashed_accounts_compared {
                println!("  ✓ Hashed account values");
            }
            if self.storage_trie_compared {
                println!("  ✓ Storage trie nodes");
            }
            return;
        }

        println!("\n✗ FAILED: {} differences found\n", self.total_differences());

        if !self.account_trie_diffs.is_empty() {
            println!("┌─ Account Trie Differences ({}) ─────────────────────────",
                self.account_trie_diffs.len());
            for (i, diff) in self.account_trie_diffs.iter().enumerate() {
                if !verbose && i >= 10 {
                    println!("  ... and {} more", self.account_trie_diffs.len() - 10);
                    break;
                }
                println!("  Path: {:?}", diff.path);
                println!("    Canonical: {}", diff.canonical_value);
                println!("    Proofs:    {}", diff.proofs_value);
                println!();
            }
        }

        if !self.hashed_account_diffs.is_empty() {
            println!("┌─ Hashed Account Differences ({}) ───────────────────────",
                self.hashed_account_diffs.len());
            for (i, diff) in self.hashed_account_diffs.iter().enumerate() {
                if !verbose && i >= 10 {
                    println!("  ... and {} more", self.hashed_account_diffs.len() - 10);
                    break;
                }
                println!("  Address: {}", diff.address);
                if let Some((canonical, proofs)) = &diff.balance_diff {
                    println!("    Balance:      canonical={}, proofs={}", canonical, proofs);
                }
                if let Some((canonical, proofs)) = &diff.nonce_diff {
                    println!("    Nonce:        canonical={}, proofs={}", canonical, proofs);
                }
                if let Some((canonical, proofs)) = &diff.code_hash_diff {
                    println!("    Code Hash:    canonical={}, proofs={}", canonical, proofs);
                }
                if let Some((canonical, proofs)) = &diff.storage_root_diff {
                    println!("    Storage Root: canonical={}, proofs={}", canonical, proofs);
                }
                println!();
            }
        }

        if !self.storage_trie_diffs.is_empty() {
            println!("┌─ Storage Trie Differences ({}) ─────────────────────────",
                self.storage_trie_diffs.len());
            for (i, diff) in self.storage_trie_diffs.iter().enumerate() {
                if !verbose && i >= 10 {
                    println!("  ... and {} more", self.storage_trie_diffs.len() - 10);
                    break;
                }
                println!("  Hashed Address: {:?}, Path: {:?}", diff.hashed_address, diff.path);
                println!("    Canonical: {}", diff.canonical_value);
                println!("    Proofs:    {}", diff.proofs_value);
                println!();
            }
        }

        if !self.storage_diffs.is_empty() {
            println!("┌─ Storage Differences ({}) ──────────────────────────────",
                self.storage_diffs.len());
            for (i, diff) in self.storage_diffs.iter().enumerate() {
                if !verbose && i >= 10 {
                    println!("  ... and {} more", self.storage_diffs.len() - 10);
                    break;
                }
                println!("  Hashed Address: {:?}, Slot: {}", diff.hashed_address, diff.slot);
                println!("    Canonical: {}", diff.canonical_value);
                println!("    Proofs:    {}", diff.proofs_value);
                println!();
            }
        }

        println!("─────────────────────────────────────────────────────────────────\n");
    }
}

#[derive(Debug)]
struct AccountTrieDiff {
    path: Nibbles,
    canonical_value: String,
    proofs_value: String,
}

#[derive(Debug)]
struct HashedAccountDiff {
    address: B256,
    balance_diff: Option<(U256, U256)>,
    nonce_diff: Option<(u64, u64)>,
    code_hash_diff: Option<(B256, B256)>,
    storage_root_diff: Option<(B256, B256)>,
}

#[derive(Debug)]
struct StorageTrieDiff {
    hashed_address: B256,
    path: Nibbles,
    canonical_value: String,
    proofs_value: String,
}

#[derive(Debug)]
struct StorageDiff {
    hashed_address: B256,
    slot: B256,
    canonical_value: U256,
    proofs_value: U256,
}

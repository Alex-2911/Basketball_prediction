@@
     shortlist.to_csv(shortlist_path, index=False, encoding="utf-8")
     logging.info("Saved bet shortlist -> %s (%d rows)", shortlist_path, len(shortlist))
@@
     logging.info(
         "[LOCAL_MATCHED] verification live_shortlist_rows=%d resolved_historical_rows=%d export_rows=%d",
         int(len(shortlist)),
         int(len(historical_subset_for_export)),
         int(len(matched_export_latest)),
     )
+
+    # --- Persist Script 11 watchlist history (only after full enrichment) ---
+    if persist_script11_watchlist_history is not None:
+        try:
+            combined_path_for_history = Path(pred_dir) / f"combined_nba_predictions_acc_{requested_ymd}.csv"
+
+            required_enriched_cols = [
+                "prob_live_oos_proxy",
+                "prob_live_safe_pre_clip",
+                "prob_base",
+                "prob_used",
+                "market_implied_p_raw",
+                "market_implied_p_devig",
+                "model_market_gap",
+                "model_market_gap_flag",
+                "live_underdog_upscale_guard_triggered",
+                "live_shrink_triggered",
+                "blocked_by",
+                "EV_base_€_per_100",
+                "EV_live_€_per_100",
+                "EV_€_per_100",
+                "home_win_rate",
+                "rules_passed",
+                "margin_hw",
+                "margin_odds",
+                "margin_prob",
+                "margin_ev",
+            ]
+
+            def _has_enrichment(df: pd.DataFrame) -> bool:
+                if df is None or df.empty:
+                    return False
+                cols_ok = all(c in df.columns for c in required_enriched_cols)
+                nonnull_key = any(
+                    (c in df.columns and df[c].notna().any())
+                    for c in ("prob_base", "prob_used", "EV_€_per_100", "EV_live_€_per_100")
+                )
+                return cols_ok and nonnull_key
+
+            # Persist fully enriched upcoming rows (df_future) as upcoming_d
+            if "df_future" in locals() and _has_enrichment(df_future):
+                persist_script11_watchlist_history(
+                    rows_df=df_future.copy(),
+                    output_dir=out_dir,
+                    run_date=requested_ymd,
+                    params_used=params_for_eval if "params_for_eval" in locals() else None,
+                    chosen=(selection_decision if "selection_decision" in locals() else None),
+                    compareN=(int(max(N_WINDOWS)) if N_WINDOWS else None),
+                    combined_predictions_path=str(combined_path_for_history),
+                    source="upcoming_d",
+                )
+                logging.info("Persisted enriched upcoming_d -> script11_watchlist_history (rows=%d)", len(df_future))
+            else:
+                logging.info("Skipping persist of upcoming_d: enrichment columns missing or no upcoming rows.")
+
+            # Persist dedicated watchlist if it exists (df_w / watchlist)
+            watchlist_candidates = []
+            for v in ("df_w", "watchlist", "df_watchlist", "watchlist_df"):
+                if v in locals() and isinstance(locals()[v], pd.DataFrame) and not locals()[v].empty:
+                    watchlist_candidates.append(locals()[v])
+
+            if watchlist_candidates:
+                for wi, wdf in enumerate(watchlist_candidates):
+                    if _has_enrichment(wdf):
+                        persist_script11_watchlist_history(
+                            rows_df=wdf.copy(),
+                            output_dir=out_dir,
+                            run_date=requested_ymd,
+                            params_used=params_for_eval if "params_for_eval" in locals() else None,
+                            chosen=(selection_decision if "selection_decision" in locals() else None),
+                            compareN=(int(max(N_WINDOWS)) if N_WINDOWS else None),
+                            combined_predictions_path=str(combined_path_for_history),
+                            source="watchlist",
+                        )
+                        logging.info("Persisted dedicated watchlist candidate #%d -> script11_watchlist_history (rows=%d)", wi+1, len(wdf))
+                    else:
+                        logging.info("Skipping persist of watchlist candidate #%d: enrichment columns incomplete", wi+1)
+            else:
+                # Optional: persist shortlist as watchlist only if it contains enrichment columns
+                if "shortlist" in locals() and isinstance(shortlist, pd.DataFrame) and not shortlist.empty:
+                    if any(col in shortlist.columns for col in ["prob_base", "prob_used", "live_shrink_triggered", "blocked_by"]):
+                        if _has_enrichment(shortlist):
+                            persist_script11_watchlist_history(
+                                rows_df=shortlist.copy(),
+                                output_dir=out_dir,
+                                run_date=requested_ymd,
+                                params_used=params_for_eval if "params_for_eval" in locals() else None,
+                                chosen=(selection_decision if "selection_decision" in locals() else None),
+                                compareN=(int(max(N_WINDOWS)) if N_WINDOWS else None),
+                                combined_predictions_path=str(combined_path_for_history),
+                                source="watchlist",
+                            )
+                            logging.info("Persisted shortlist as watchlist (rows=%d)", len(shortlist))
+                        else:
+                            logging.info("Shortlist present but missing enrichment columns; not persisted as watchlist.")
+                else:
+                    logging.info("No dedicated watchlist (df_w) found; not persisting watchlist.")
+        except Exception as _e:
+            logging.warning("persist_script11_watchlist_history failed: %s", _e)
+    else:
+        logging.info("persist_script11_watchlist_history helper not available; skipping persistence.")
@@
     write_json(metrics_snapshot_path, snapshot)
     write_json(metrics_snapshot_dated_path, snapshot)
@@
 if __name__ == "__main__":
     main()

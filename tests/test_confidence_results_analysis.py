from experiments.analyze_confidence_merge_results import stability_summary


def test_stability_summary_accepts_live_and_csv_rows():
    rows = [dict(method='confidence', family='noise', count_changed=True,
                 ARI=.8, changed_merge_links=4),
            dict(method='confidence', family='noise', count_changed='False',
                 ARI='1.0', changed_merge_links='0')]
    summary = stability_summary(rows)
    assert summary == [dict(method='confidence', family='noise', runs=2,
                            count_changes=1, mean_ARI=.9, changed_merge_links=4)]

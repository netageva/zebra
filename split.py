import pandas as pd
import config as CFG

all_reports = pd.read_csv(CFG.CSV)
test_df = all_reports.iloc[CFG.TEST_IND]['text']

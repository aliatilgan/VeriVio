import pandas as pd
import sys
sys.path.append('backend')
from modules.hypothesis_testing.manova import run_manova

# Test verisini yükle
df = pd.read_csv('test_data.csv')
print('MANOVA testi başlıyor...')
print('Veri boyutu:', df.shape)
print('Sütunlar:', df.columns.tolist())

try:
    result = run_manova(df, ['y1', 'y2'], 'group')
    print('✅ MANOVA başarılı!')
    print('Test sonuçları:', list(result['test_results'].keys()))
    if 'Wilks\' lambda' in result['test_results']:
        wilks = result['test_results']['Wilks\' lambda']
        print(f'Wilks Lambda: {wilks["statistic"]:.4f}, p-value: {wilks["p_value"]:.4f}')
    else:
        print('Test sonuçları:', result['test_results'])
except Exception as e:
    print('❌ MANOVA hatası:', str(e))
    import traceback
    traceback.print_exc()
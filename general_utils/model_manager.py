from pathlib import Path
import joblib
import pandas as pd
from datetime import datetime

def save_model_package(model, model_name, best_params, metrics, save_dir='./models'):
    save_path = Path(save_dir)
    
    save_path.mkdir(parents=True, exist_ok=True)
        
    package = {
        'model_name': model_name,
        'model': model, 
        'best_params': best_params,
        'metrics': metrics,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    file_name = f"{model_name.replace(' ', '_')}.pkl"
    file_path = save_path / file_name
    
    joblib.dump(package, file_path)
    
    return str(file_path)


def generate_report(model_dir='./models'):
    results = []
    path = Path(model_dir)
    
    if not path.is_dir():
        print(f"Folder {path.resolve()} không tồn tại hoặc không có model nào cả")
        return pd.DataFrame()
        
    for file_path in path.glob('*.pkl'):
        if file_path.name.startswith(('onehot', 'standard', 'tfidf', 'vectorizer')):
            continue
            
        try:
            package = joblib.load(file_path)
            
            row = {
                'Mô hình': package.get('model_name', 'Unknown'),
                'Tham số tốt nhất': str(package.get('best_params', {}))
            }
            
            metrics = package.get('metrics', {})
            for metric_name, value in metrics.items():
                row[metric_name] = value
                
            results.append(row)
        except Exception as e:
            print(f"-> Skip file {file_path.name} vì lỗi cấu trúc gói. Chi tiết: {e}")
                
    df_leaderboard = pd.DataFrame(results)
    
    if not df_leaderboard.empty:
        if 'F1_Score' in df_leaderboard.columns:
            df_leaderboard['F1_Score'] = pd.to_numeric(df_leaderboard['F1_Score'])
            df_leaderboard = df_leaderboard.sort_values(by='F1_Score', ascending=False).reset_index(drop=True)
            
        elif 'MAE' in df_leaderboard.columns:
            df_leaderboard['MAE'] = pd.to_numeric(df_leaderboard['MAE'])
            df_leaderboard = df_leaderboard.sort_values(by='MAE', ascending=True).reset_index(drop=True)

        if 'Tham số tốt nhất' in df_leaderboard.columns:
            cols = [c for c in df_leaderboard.columns if c != 'Tham số tốt nhất'] + ['Tham số tốt nhất']
            df_leaderboard = df_leaderboard[cols]
            
    return df_leaderboard
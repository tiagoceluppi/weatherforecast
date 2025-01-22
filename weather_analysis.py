import numpy as np
import pandas as pd
import cupy as cp
import matplotlib.pyplot as plt
import psycopg2
from datetime import datetime
import os
from scipy import stats

def get_weather_data(db_params, days=365):
    """Carrega dados do banco"""
    query = """
    SELECT 
        m.temperatura_max,
        m.temperatura_min,
        m.precipitacao_total,
        m.umidade_rel_hora,
        m.pressao_atm_hora,
        m.vento_velocidade,
        m.radiacao_global,
        m.data,
        e.id_municipio,
        e.latitude,
        e.longitude,
        e.altitude
    FROM microdados m
    JOIN estacao e ON m.id_estacao = e.id_estacao
    WHERE m.data >= CURRENT_DATE - INTERVAL '%s days'
    """
    
    try:
        conn = psycopg2.connect(**db_params)
        print("Carregando dados do banco...")
        df = pd.read_sql_query(query, conn, params=[days])
        print(f"✓ Dados carregados: {len(df)} registros de {df['id_municipio'].nunique()} municípios")
        conn.close()
        return df
    except Exception as e:
        print(f"✗ Erro ao carregar dados: {e}")
        return None

def generate_histograms(df):
    """Gera histogramas para todas as variáveis numéricas"""
    variables = {
        'temperatura_max': 'Temperatura Máxima (°C)',
        'temperatura_min': 'Temperatura Mínima (°C)',
        'precipitacao_total': 'Precipitação Total (mm)',
        'umidade_rel_hora': 'Umidade Relativa (%)',
        'pressao_atm_hora': 'Pressão Atmosférica (hPa)',
        'vento_velocidade': 'Velocidade do Vento (m/s)',
        'radiacao_global': 'Radiação Global (kJ/m²)'
    }
    
    plt.style.use('default')
    rows = (len(variables) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(15, 4*rows))
    axes = axes.flatten()
    
    stats_dict = {}
    
    for idx, (var, label) in enumerate(variables.items()):
        if var in df.columns:
            data = df[var].dropna()
            
            try:
                data_gpu = cp.array(data)
                mean = float(cp.mean(data_gpu))
                std = float(cp.std(data_gpu))
                min_val = float(cp.min(data_gpu))
                max_val = float(cp.max(data_gpu))
                
                hist, bins = cp.histogram(data_gpu, bins=50)
                hist = cp.asnumpy(hist)
                bins = cp.asnumpy(bins)
                
                axes[idx].hist(data, bins=50, alpha=0.7, color='skyblue')
                axes[idx].set_title(label, fontsize=12, pad=10)
                axes[idx].grid(True, alpha=0.3)
                axes[idx].set_xlabel('Valor')
                axes[idx].set_ylabel('Frequência')
                
                stats_text = (f'Média: {mean:.2f}\n'
                            f'DP: {std:.2f}\n'
                            f'Mín: {min_val:.2f}\n'
                            f'Máx: {max_val:.2f}')
                
                axes[idx].text(0.95, 0.95, stats_text,
                             transform=axes[idx].transAxes,
                             verticalalignment='top',
                             horizontalalignment='right',
                             bbox=dict(boxstyle='round', 
                                     facecolor='white', 
                                     alpha=0.8,
                                     edgecolor='gray'))
                
                density = stats.gaussian_kde(data)
                xs = np.linspace(min_val, max_val, 200)
                axes[idx].plot(xs, density(xs) * len(data) * (bins[1] - bins[0]), 
                             'r-', lw=2, alpha=0.7)
                
                stats_dict[var] = {
                    'media': mean,
                    'desvio_padrao': std,
                    'minimo': min_val,
                    'maximo': max_val,
                    'registros': len(data),
                    'dados_faltantes': df[var].isnull().sum()
                }
                
            except Exception as e:
                print(f"Erro ao processar {var}: {e}")
    
    for idx in range(len(variables), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    stats_df = pd.DataFrame.from_dict(stats_dict, orient='index')
    stats_df.index.name = 'Variável'
    stats_df.columns = ['Média', 'Desvio Padrão', 'Mínimo', 'Máximo', 
                       'Total Registros', 'Dados Faltantes']
    
    return fig, stats_df

def main():
    # Configurações do banco
    db_params = {
		'dbname': 'system_climasim_db',
		'user': 'system_climasim_user',
		'password': '',
		'host': '192.168.0.127'
    }
    
    # Carrega dados
    df = get_weather_data(db_params)
    if df is None:
        return
    
    # Gera análises
    print("\nGerando histogramas e estatísticas...")
    fig, stats_df = generate_histograms(df)
    
    # Cria diretório para outputs
    output_dir = 'analise_meteorologica'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Salva resultados
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig.savefig(f'{output_dir}/histogramas_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    stats_df.to_csv(f'{output_dir}/estatisticas_{timestamp}.csv')
    
    # Mostra resultados
    print("\nEstatísticas das variáveis:")
    print(stats_df.round(2))
    
    print(f"\nArquivos salvos em:")
    print(f"- {output_dir}/histogramas_{timestamp}.png")
    print(f"- {output_dir}/estatisticas_{timestamp}.csv")
    
    plt.show()

if __name__ == "__main__":
    main()

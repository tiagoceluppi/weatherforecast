# weather_forecast_cuda.py
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from numba import cuda
import math
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def load_historical_data(db_params):
    """
    Carrega dados históricos para base da previsão
    """
    query = """
    WITH sample_data AS (
        SELECT 
            m.temperatura_max,
            m.temperatura_min,
            m.precipitacao_total,
            m.umidade_rel_hora,
            m.pressao_atm_hora,
            m.vento_velocidade,
            m.radiacao_global,
            m.data,
            e.latitude,
            e.longitude,
            EXTRACT(MONTH FROM m.data) as mes
        FROM microdados m
        JOIN estacao e ON m.id_estacao = e.id_estacao
        WHERE 
            m.temperatura_max IS NOT NULL
            AND m.temperatura_min IS NOT NULL
            AND m.precipitacao_total IS NOT NULL
            AND m.vento_velocidade IS NOT NULL
        ORDER BY m.data DESC
        LIMIT 1000
    )
    SELECT * FROM sample_data;
    """
    
    try:
        engine = create_engine(f'postgresql://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}/{db_params["dbname"]}')
        print("Carregando dados meteorológicos...")
        df = pd.read_sql_query(query, engine)
        
        if len(df) == 0:
            print("✗ Nenhum dado encontrado no banco!")
            return None
            
        print(f"✓ {len(df)} registros carregados")
        print(f"✓ Período: {df['data'].min()} a {df['data'].max()}")
        
        # Estatísticas básicas
        print("\nEstatísticas das variáveis:")
        stats = df[['temperatura_max', 'temperatura_min', 'precipitacao_total', 'vento_velocidade']].describe()
        print(stats.round(2))
        
        return df, stats

    except Exception as e:
        print(f"✗ Erro ao carregar dados: {e}")
        return None, None

@cuda.jit
def weather_forecast_kernel(input_data, seasonal_patterns, stat_limits, output_temp_max, 
                          output_temp_min, output_precip, output_wind):
    """
    Kernel CUDA para previsão meteorológica com limites realistas
    """
    idx = cuda.grid(1)
    if idx < 5:  # 5 dias de previsão
        # Limites estatísticos
        temp_min_limit = stat_limits[0]  # min temperatura
        temp_max_limit = stat_limits[1]  # max temperatura
        precip_max = stat_limits[2]      # max precipitação
        wind_max = stat_limits[3]        # max vento
        
        # Base para temperaturas
        temp_max_base = 0.0
        temp_min_base = 0.0
        precip_base = 0.0
        wind_base = 0.0
        
        # Média ponderada dos últimos 3 dias
        for i in range(3):
            weight = 0.5 if i == 0 else 0.3 if i == 1 else 0.2
            temp_max_base += input_data[i, 0] * weight
            temp_min_base += input_data[i, 1] * weight
            precip_base += input_data[i, 2] * weight
            wind_base += input_data[i, 3] * weight
        
        # Variação diária (±2°C)
        day_variation = math.sin(idx * 0.5) * 2.0
        
        # Aplica previsões com limites
        output_temp_max[idx] = max(temp_min_limit, 
                                 min(temp_max_limit, 
                                     temp_max_base + day_variation))
        
        output_temp_min[idx] = max(temp_min_limit,
                                 min(output_temp_max[idx] - 2.0,
                                     temp_min_base + day_variation))
        
        # Precipitação com limite realista
        output_precip[idx] = max(0.0, 
                                min(precip_max,
                                    precip_base + (math.sin(idx * 0.7) * 2.0)))
        
        # Vento com limite realista
        output_wind[idx] = max(0.0,
                              min(wind_max,
                                  wind_base + abs(math.cos(idx * 0.3) * 3.0)))

def generate_forecast(df, stats):
    """
    Gera previsão usando dados históricos e limites estatísticos
    """
    try:
        # Prepara dados de entrada
        input_data = np.column_stack([
            df['temperatura_max'].values[:3],
            df['temperatura_min'].values[:3],
            df['precipitacao_total'].values[:3],
            df['vento_velocidade'].values[:3]
        ]).astype(np.float32)
        
        # Padrões sazonais
        seasonal = df.groupby('mes')['temperatura_max'].mean().values.astype(np.float32)
        
        # Limites estatísticos
        stat_limits = np.array([
            stats.loc['min', 'temperatura_min'],
            stats.loc['max', 'temperatura_max'],
            stats.loc['max', 'precipitacao_total'],
            stats.loc['max', 'vento_velocidade']
        ], dtype=np.float32)
        
        # Configuração CUDA
        threadsperblock = 256
        blockspergrid = 1
        
        # Arrays GPU
        d_input = cuda.to_device(input_data)
        d_seasonal = cuda.to_device(seasonal)
        d_stat_limits = cuda.to_device(stat_limits)
        d_output_temp_max = cuda.device_array(5, dtype=np.float32)
        d_output_temp_min = cuda.device_array(5, dtype=np.float32)
        d_output_precip = cuda.device_array(5, dtype=np.float32)
        d_output_wind = cuda.device_array(5, dtype=np.float32)
        
        print("\nConfigurações CUDA:")
        print(f"✓ Threads por bloco: {threadsperblock}")
        print(f"✓ Blocos no grid: {blockspergrid}")
        
        # Executa kernel
        weather_forecast_kernel[blockspergrid, threadsperblock](
            d_input, d_seasonal, d_stat_limits,
            d_output_temp_max, d_output_temp_min,
            d_output_precip, d_output_wind
        )
        
        # Recupera resultados
        temp_max = d_output_temp_max.copy_to_host()
        temp_min = d_output_temp_min.copy_to_host()
        precip = d_output_precip.copy_to_host()
        wind = d_output_wind.copy_to_host()
        
        # Limpa memória GPU
        del d_input, d_seasonal, d_stat_limits
        del d_output_temp_max, d_output_temp_min, d_output_precip, d_output_wind
        
        return temp_max, temp_min, precip, wind
        
    except Exception as e:
        print(f"✗ Erro na previsão: {e}")
        return None, None, None, None

def get_weather_condition(temp_max, precip):
    """Determina condição do tempo"""
    if precip > 5:
        return "Chuva"
    elif precip > 0.5:
        return "Possibilidade de Chuva"
    elif temp_max > 30:
        return "Ensolarado"
    else:
        return "Parcialmente Nublado"

def format_forecast(temp_max, temp_min, precip, wind):
    """Formata a previsão para exibição"""
    forecast = []
    
    for i in range(5):
        date = datetime.now() + timedelta(days=i+1)
        condition = get_weather_condition(temp_max[i], precip[i])
        
        day_forecast = {
            'data': date.strftime('%d/%m/%Y'),
            'dia_semana': date.strftime('%A'),
            'temp_max': round(temp_max[i], 1),
            'temp_min': round(temp_min[i], 1),
            'precipitacao': round(precip[i], 1),
            'vento': round(wind[i], 1),
            'condicao': condition
        }
        forecast.append(day_forecast)
    
    return forecast

def plot_forecast(forecast):
    """Gera visualização da previsão"""
    days = [f['data'] for f in forecast]
    temp_max = [f['temp_max'] for f in forecast]
    temp_min = [f['temp_min'] for f in forecast]
    precip = [f['precipitacao'] for f in forecast]
    wind = [f['vento'] for f in forecast]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Temperaturas
    ax1.plot(days, temp_max, 'ro-', label='Máxima', linewidth=2)
    ax1.plot(days, temp_min, 'bo-', label='Mínima', linewidth=2)
    ax1.set_title('Previsão de Temperatura')
    ax1.set_ylabel('Temperatura (°C)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Precipitação
    bars = ax2.bar(days, precip, color='blue', alpha=0.6)
    ax2.set_title('Previsão de Precipitação')
    ax2.set_ylabel('Precipitação (mm)')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Vento
    ax3.plot(days, wind, 'go-', linewidth=2)
    ax3.set_title('Previsão de Vento')
    ax3.set_ylabel('Velocidade (km/h)')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('previsao_tempo.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    db_params = {
		'dbname': 'system_climasim_db',
		'user': 'system_climasim_user',
		'password': '',
		'host': '192.168.0.127'

    }
    
    # Carrega dados
    result = load_historical_data(db_params)
    if result is None:
        return
    df, stats = result
    
    # Gera previsão
    temp_max, temp_min, precip, wind = generate_forecast(df, stats)
    if temp_max is None:
        return
    
    # Formata e exibe resultados
    forecast = format_forecast(temp_max, temp_min, precip, wind)
    
    print("\nPrevisão para os próximos 5 dias:")
    print("=" * 60)
    
    for day in forecast:
        print(f"\nData: {day['data']} ({day['dia_semana']})")
        print(f"Temperatura: {day['temp_min']}°C - {day['temp_max']}°C")
        print(f"Condição: {day['condicao']}")
        print(f"Precipitação: {day['precipitacao']} mm")
        print(f"Vento: {day['vento']} km/h")
    
    # Gera gráfico
    plot_forecast(forecast)
    print("\n✓ Gráfico salvo como 'previsao_tempo.png'")

if __name__ == "__main__":
    print("Sistema de Previsão Meteorológica com CUDA")
    print("=" * 50)
    main()

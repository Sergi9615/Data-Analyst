
from googleapiclient.http import MediaIoBaseDownload
import io
import pandas as pd
import numpy as np
from prophet import Prophet

resultados_tracks = []

def run_forecast_and_collect(csv_info):
    artist = csv_info['artist']
    file_id = csv_info['file_id']
    print(f"\n📥 Procesando forecast para: {artist}")

    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)

    df = pd.read_csv(fh)
    data_clean = df.dropna(subset=['Estimated_Income', 'Track'])
    data_spotify = data_clean[data_clean['Platform'] == 'spotify'].copy()
    data_spotify['Date'] = pd.to_datetime(data_spotify['Date'], errors='coerce')
    data_spotify = data_spotify.dropna(subset=['Date'])
    data_spotify['Month'] = data_spotify['Date'].dt.to_period('M')
    song_monthly_revenue = data_spotify.groupby(['Month', 'Track'])['Estimated_Income'].sum().reset_index()
    song_monthly_revenue['Month'] = song_monthly_revenue['Month'].dt.to_timestamp()

    today = pd.Timestamp.today().to_period('M').to_timestamp()
    song_monthly_revenue = song_monthly_revenue[song_monthly_revenue['Month'] < today]
    if song_monthly_revenue.empty:
        print(f"⚠️ {artist}: No hay data usable.")
        return

    for track in song_monthly_revenue['Track'].unique():
        df_track = song_monthly_revenue[song_monthly_revenue['Track'] == track].copy()
        df_prophet = df_track.rename(columns={'Month': 'ds', 'Estimated_Income': 'y'})[['ds', 'y']]

        if df_prophet['y'].sum() == 0 or (df_prophet['y'] == 0).mean() > 0.9 or len(df_prophet) < 6:
            continue

        ingreso_historico_total = df_prophet['y'].sum()

        modelo = Prophet(
            yearly_seasonality=len(df_prophet) >= 12,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.3 if len(df_prophet) >= 12 else 0.05,
            n_changepoints=30 if len(df_prophet) >= 12 else 15
        )
        modelo.fit(df_prophet)

        future = modelo.make_future_dataframe(periods=12, freq='MS')
        forecast = modelo.predict(future)

        primer_mes = df_prophet['ds'].max() + pd.offsets.MonthBegin(1)
        segundo_mes = primer_mes + pd.offsets.MonthBegin(1)

        ultimos_3 = df_prophet.tail(3)
        if len(ultimos_3) == 3:
            x = np.arange(3)
            y = ultimos_3['y'].values
            pendiente = np.polyfit(x, y, 1)[0]
            idx = forecast[forecast['ds'] == primer_mes].index
            if not idx.empty:
                i = idx[0]
                y_last = df_prophet['y'].iloc[-1]
                forecast.at[i, 'yhat'] = y_last if abs(pendiente) < 0.05 else min(y_last + pendiente * 1.5, y_last * 1.5)

        idx2 = forecast[forecast['ds'] == segundo_mes].index
        if not idx2.empty:
            i = idx2[0]
            forecast.at[i, 'yhat'] = df_prophet['y'].tail(3).mean()

        yhat_values = forecast['yhat'].values.copy()
        for i in range(1, len(yhat_values)):
            max_growth = yhat_values[i - 1] * 1.3
            salto = yhat_values[i] - yhat_values[i - 1]
            std_previa = np.std(yhat_values[max(i - 6, 0):i])
            if yhat_values[i] > max_growth and salto > std_previa * 1.5:
                yhat_values[i] = max_growth
        forecast['yhat'] = yhat_values

        forecast_2025 = forecast[(forecast['ds'] >= '2025-01-01') & (forecast['ds'] <= '2025-12-01')]
        suma_2025 = forecast_2025['yhat'].sum()

        if suma_2025 < 0:
            continue

        crecimiento = (suma_2025 - ingreso_historico_total) / ingreso_historico_total * 100 if ingreso_historico_total > 0 else 0

        resultados_tracks.append({
            'Artist': artist,
            'Track': track,
            'Meses_históricos': len(df_prophet),
            'Ingreso_historico_total': round(ingreso_historico_total, 2),
            'Ingreso_proyectado_2025': round(suma_2025, 2),
            'Crecimiento_%': round(crecimiento, 2)
        })

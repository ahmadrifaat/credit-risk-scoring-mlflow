import time
import random
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary

#10 metriks

# 1. Total request inferensi yang masuk
REQUEST_COUNT = Counter('model_prediction_requests_total', 'Total number of prediction requests')

# 2. Total prediksi sukses
PREDICTION_SUCCESS = Counter('model_prediction_success_total', 'Total successful predictions')

# 3. Total prediksi gagal / error
PREDICTION_ERROR = Counter('model_prediction_error_total', 'Total failed predictions')

# 4. Total nasabah berisiko tinggi (Class 1) yang terdeteksi
HIGH_RISK_DETECTED = Counter('model_high_risk_detected_total', 'Total high risk customers detected')

# 5. Total nasabah berisiko rendah (Class 0) yang terdeteksi
LOW_RISK_DETECTED = Counter('model_low_risk_detected_total', 'Total low risk customers detected')

# 6. Durasi waktu proses inferensi model (Latency)
PREDICTION_LATENCY = Histogram('model_prediction_latency_seconds', 'Time spent processing prediction')

# 7. Penggunaan Memory RAM simulasi aplikasi (Gauge bisa naik turun)
MEMORY_USAGE_GB = Gauge('app_memory_usage_gigabytes', 'Simulated RAM usage of the application')

# 8. Penggunaan CPU simulasi aplikasi
CPU_USAGE_PERCENT = Gauge('app_cpu_usage_percent', 'Simulated CPU usage percentage')

# 9. Jumlah request aktif yang sedang berjalan saat ini
ACTIVE_REQUESTS = Gauge('app_active_requests', 'Number of active requests currently being processed')

# 10. Ringkasan ukuran data input (jumlah fitur/kolom) yang dikirim nasabah
INPUT_FEATURE_SIZE_SUMMARY = Summary('model_input_features_size', 'Summary of input feature sizes received')


# simulasi
def simulate_model_inference():
    while True:
        # latensi dan req
        ACTIVE_REQUESTS.inc()
        start_time = time.time()
        
        # total req masuk
        REQUEST_COUNT.inc()
        
        # simulasi fitur input
        INPUT_FEATURE_SIZE_SUMMARY.observe(4)
        
        #simulasi performa
        MEMORY_USAGE_GB.set(random.uniform(2.1, 4.8))
        CPU_USAGE_PERCENT.set(random.uniform(15.0, 85.0))
        
        # simulasi error prob
        if random.random() < 0.02:
            PREDICTION_ERROR.inc()
            time.sleep(random.uniform(0.5, 2.0)) # Delay error
        else:
            PREDICTION_SUCCESS.inc()
            
            # simulasi hasil
            prediction_result = random.choice([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
            if prediction_result == 1:
                HIGH_RISK_DETECTED.inc()
            else:
                LOW_RISK_DETECTED.inc()
                
            time.sleep(random.uniform(0.01, 0.3)) # Delay inferensi normal
            
        #latensi akhir
        PREDICTION_LATENCY.observe(time.time() - start_time)
        ACTIVE_REQUESTS.dec()
        
        print("Mengekspor metrik ke Prometheus... (Tekan Ctrl+C untuk berhenti)")
        time.sleep(1)

if __name__ == '__main__':
    # exporter lokal
    PORT = 8000
    start_http_server(PORT)
    print(f"Prometheus Exporter ready! Berjalan di http://localhost:{PORT}/metrics")
    simulate_model_inference()
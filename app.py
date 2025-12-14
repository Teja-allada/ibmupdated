import os
import time
import cv2
import numpy as np
import streamlit as st
from enhanced_test import TrafficMonitor

# Page configuration
st.set_page_config(
    page_title="Enhanced Traffic Monitoring UI",
    page_icon="🚦",
    layout="wide",
)

# Custom styling: traffic security theme
st.markdown(
    """
    <style>
    :root {
      --bg: #0b1220;
      --panel: #0f172a;
      --text: #e5e7eb;
      --danger: #ef4444;
      --warn: #f59e0b;
      --safe: #22c55e;
      --accent: #3b82f6;
    }
    body { background: linear-gradient(180deg, var(--bg) 0%, #111827 100%); }
    .block-container { padding-top: 0.75rem; }
    [data-testid="stSidebar"] { background-color: #0c1324; }
    [data-testid="stSidebar"] .stButton>button { background: linear-gradient(90deg, var(--accent), var(--warn)); color: white; border: none; }
    h1, h2, h3 { color: var(--text); }
    .security-banner { 
      margin: 0.5rem 0 1rem; padding: 0.75rem 1rem; border-radius: 12px;
      background: linear-gradient(90deg, var(--danger), var(--warn), var(--safe));
      color: #111827; font-weight: 700; letter-spacing: 0.2px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }
    .metric-box { background: var(--panel); padding: 1rem; border-radius: 12px; border: 1px solid #1f2937; }
    [data-testid="stMetric"] { background: var(--panel); padding: 0.75rem; border-radius: 12px; border-left: 4px solid var(--warn); }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--text); }
    [data-testid="stMetric"] [data-testid="stMetricLabel"] { color: #9ca3af; }
    .stButton>button { 
      background: linear-gradient(90deg, var(--safe), var(--accent));
      color: white; border-radius: 10px; padding: 0.5rem 1rem; border: none;
      box-shadow: 0 6px 16px rgba(34,197,94,0.25);
    }
    div[data-testid="stImage"] img { 
      border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.35);
      border: 3px solid; border-image: linear-gradient(90deg, var(--danger), var(--warn), var(--safe)) 1;
    }
    .st-emotion-cache-1kyxreq { padding-top: 0rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "running" not in st.session_state:
    st.session_state.running = False
if "monitor" not in st.session_state:
    st.session_state.monitor = None
if "video_path" not in st.session_state:
    st.session_state.video_path = None

# Banner under title
st.title("🚦 Enhanced Traffic Monitoring System")
st.markdown(
    """
    <div class="security-banner">🚧 Traffic Security Dashboard • Live Monitoring • Road Safety</div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("🛡️ Configuration")
    source_type = st.radio("Video Source", ["Webcam", "Video Path", "Upload Video"], index=0)

    video_source = 0
    if source_type == "Video Path":
        video_source = st.text_input("Enter video file path", value="", placeholder="C:/path/to/video.mp4")
    elif source_type == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
        if uploaded_file is not None:
            temp_path = os.path.join(os.getcwd(), "uploaded_video.mp4")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            video_source = temp_path

    model_path = st.text_input("Model path", value="yolo11n.pt")
    class_file = st.text_input("Classes file", value="coco.names")
    fps_override = st.number_input("FPS override (optional)", value=0, min_value=0, max_value=240, step=1)
    conf_threshold = st.slider("Detection confidence", 0.1, 0.9, 0.3, 0.05)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_clicked = st.button("▶️ Start Monitoring", use_container_width=True)
    with col_btn2:
        stop_clicked = st.button("⏹ Stop", use_container_width=True)

    st.markdown("---")
    save_analytics = st.button("💾 Save Analytics Now", use_container_width=True)
    reset_calibration = st.button("🔄 Reset Calibration", use_container_width=True)

# Handle control actions
if start_clicked and not st.session_state.running:
    # Create monitor instance lazily
    try:
        kwargs = {
            "video_source": video_source if video_source != 0 else 0,
            "model_path": model_path,
            "class_file": class_file,
        }
        if fps_override and int(fps_override) > 0:
            kwargs["fps_override"] = int(fps_override)
        st.session_state.monitor = TrafficMonitor(**kwargs)
        st.session_state.monitor.conf_threshold = float(conf_threshold)
        st.session_state.running = True
        st.success("Monitoring started")
    except Exception as e:
        st.session_state.running = False
        st.session_state.monitor = None
        st.error(f"Failed to start monitoring: {e}")

if stop_clicked and st.session_state.running:
    st.session_state.running = False
    if st.session_state.monitor:
        try:
            st.session_state.monitor.cap.release()
            st.session_state.monitor.video_writer.release()
        except Exception:
            pass
    st.session_state.monitor = None
    st.warning("Monitoring stopped")

if reset_calibration and st.session_state.monitor:
    st.session_state.monitor.is_calibrated = False
    st.session_state.monitor.pixels_per_meter = 30
    st.info("Calibration reset")

if save_analytics and st.session_state.monitor:
    st.session_state.monitor.save_final_analytics()
    st.success("Analytics saved to CSV and JSON in project folder")

# Layout for video and analytics
video_col, stats_col = st.columns([2, 1])
video_placeholder = video_col.empty()

with stats_col:
    st.subheader("📊 Real-time Stats")
    metrics_placeholder = st.empty()
    vehicle_dist_placeholder = st.empty()
    speed_dist_placeholder = st.empty()

# Main processing loop
if st.session_state.running and st.session_state.monitor:
    monitor = st.session_state.monitor
    # Render loop
    frame_count = 0
    while st.session_state.running:
        success, frame = monitor.cap.read()
        if not success:
            st.warning("End of video stream or cannot access source")
            st.session_state.running = False
            break

        processed_frame = monitor.process_frame(frame)
        rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb, channels="RGB", use_container_width=True)

        # Stats update
        stats = monitor.data_exporter.get_real_time_stats()
        with stats_col:
            if stats:
                with metrics_placeholder.container():
                    total = int(stats.get("total_vehicles", 0))
                    avg_speed = float(stats.get("avg_speed", 0.0))
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Total Vehicles (recent)", total)
                    with m2:
                        st.metric("Avg Speed (km/h)", f"{avg_speed:.1f}")
                vehicle_dist = stats.get("vehicle_distribution", {})
                if vehicle_dist:
                    vehicle_dist_placeholder.bar_chart(vehicle_dist)
                speed_dist = stats.get("speed_distribution", {})
                if speed_dist:
                    speed_dist_placeholder.bar_chart(speed_dist)

        frame_count += 1
        # Small sleep to allow UI to refresh without overloading CPU
        time.sleep(0.001)

    # Cleanup when stream ends
    try:
        monitor.cap.release()
        monitor.video_writer.release()
    except Exception:
        pass
    st.session_state.monitor = None

st.caption("Tip: Use the sidebar to configure sources, model, and actions. Save analytics anytime.")
#修改字号大小：演示结果导出  实验数据预览
#实验数据预览小标题修改成：左/mm，右/mm
#解决手机端界面上方空白问题
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from matplotlib import gridspec
import io
import warnings
import random
from datetime import datetime, timedelta
from PIL import ImageDraw, ImageFont, Image
import pandas as pd

warnings.filterwarnings("ignore")

# ===================== 智能字体兼容 =====================
font_path = "SourceHanSansCN-Regular1.otf"
try:
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.sans-serif"] = ["Source Han Sans CN"]
except:
    plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]

plt.rcParams["axes.unicode_minus"] = False
plt.switch_backend("Agg")
# ========================================================

# 常量定义
LAMBDA_M = 589.3e-9
LAMBDA_NM = 589.3
LAMBDA_MM = LAMBDA_NM * 1e-6

# 环数定义
RECORD_N = [32, 30, 28, 26, 24, 22, 20, 18, 16, 14]
RIGHT_RECORD_N = [14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

# 实验参数
MAX_USES_PER_HOUR = 3
ZOOM_STEP = 0.15
ZOOM_MIN = 0.3
ZOOM_MAX = 2.0
ZOOM_DEFAULT = 1.0
DEFAULT_H = 0.0
DEFAULT_FRINGE_PHASE = 0.0
OPTICAL_OFFSET_MM = 2.50000 * 1e6

# 误差参数
READING_ERROR_MM = 0.01
INSTRUMENT_ERROR_MM = 0.005


def add_measurement_error(value: float) -> float:
    """添加随机读数误差"""
    error = random.uniform(-READING_ERROR_MM, READING_ERROR_MM)
    return round(value + error, 3)


def check_usage_limit(mode: str) -> bool:
    """检查使用次数限制"""
    now = datetime.now()
    key = f"usage_records_{mode}"
    if key not in st.session_state:
        st.session_state[key] = []
    valid_records = [t for t in st.session_state[key] if now - t < timedelta(hours=1)]
    st.session_state[key] = valid_records
    return len(valid_records) < MAX_USES_PER_HOUR


def add_usage_record(mode: str) -> None:
    """添加使用记录"""
    key = f"usage_records_{mode}"
    if key not in st.session_state:
        st.session_state[key] = []
    st.session_state[key].append(datetime.now())


def get_remaining_uses(mode: str) -> int:
    """获取剩余使用次数"""
    now = datetime.now()
    key = f"usage_records_{mode}"
    if key not in st.session_state:
        return MAX_USES_PER_HOUR
    valid_records = [t for t in st.session_state[key] if now - t < timedelta(hours=1)]
    return max(0, MAX_USES_PER_HOUR - len(valid_records))


def check_demo_export_quota() -> bool:
    return check_usage_limit("demo_export")


def consume_demo_export_quota() -> None:
    add_usage_record("demo_export")


def get_demo_export_remaining() -> int:
    return get_remaining_uses("demo_export")


def check_sim_complete_quota() -> bool:
    return check_usage_limit("sim_complete")


def consume_sim_complete_quota() -> None:
    add_usage_record("sim_complete")


def get_sim_complete_remaining() -> int:
    return get_remaining_uses("sim_complete")


def reset_all_to_default() -> None:
    """重置所有状态"""
    old_vars = ['current_order', 'total_shifts', 'order_limit']
    for var in old_vars:
        if var in st.session_state:
            del st.session_state[var]

    st.session_state.h = DEFAULT_H
    st.session_state.fringe_count = 0
    st.session_state.left_pos = {n: None for n in RECORD_N}
    st.session_state.right_pos = {n: None for n in RIGHT_RECORD_N}
    st.session_state.drift_seed = random.uniform(-1, 1)
    st.session_state.drift_direction = random.choice([-1, 1])
    st.session_state.zoom_scale = ZOOM_DEFAULT
    st.session_state.fringe_phase = DEFAULT_FRINGE_PHASE
    st.session_state.total_shift = 0.0
    st.session_state.experiment_completed = False
    st.session_state.need_reset = True
    st.session_state.reset_counter = st.session_state.get('reset_counter', 0) + 1


def calculate_uncertainty() -> dict | None:
    """计算不确定度"""
    try:
        left_vals = st.session_state.left_pos
        right_vals = st.session_state.right_pos

        r_mm = []
        for n in RECORD_N:
            l_val = left_vals[n]
            r_val = right_vals[n]
            if l_val is None or r_val is None:
                return None
            avg_r = (abs(l_val) + abs(r_val)) / 2.0
            r_mm.append(avg_r)

        delta_K = 10
        D_mm = [2 * r for r in r_mm]
        deltas_D2 = []
        for i in range(5):
            deltas_D2.append(D_mm[i]**2 - D_mm[i+5]**2)

        mean_delta_D2 = sum(deltas_D2) / len(deltas_D2)
        R_measured_mm = mean_delta_D2 / (4 * delta_K * LAMBDA_MM)
        R_theory_mm = st.session_state.R_mm
        rel_error = abs(R_measured_mm - R_theory_mm) / R_theory_mm * 100

        sum_sq = sum((d - mean_delta_D2) ** 2 for d in deltas_D2)
        sigma = np.sqrt(sum_sq / (len(deltas_D2) - 1))
        u_A_D2 = sigma / np.sqrt(len(deltas_D2))
        u_A = u_A_D2 / (4 * delta_K * LAMBDA_MM)
        u_B = INSTRUMENT_ERROR_MM / np.sqrt(3)
        u_C = np.sqrt(u_A**2 + u_B**2)

        return {
            "lambda_true": LAMBDA_NM,
            "R_theory": round(R_theory_mm, 3),
            "R_measured": round(R_measured_mm, 3),
            "rel_error": round(rel_error, 2),
            "u_A": round(u_A, 3),
            "u_B": round(u_B, 5),
            "u_C": round(u_C, 3),
        }
    except Exception:
        return None


def calculate_newtons_rings(scale: float = 1.0) -> np.ndarray:
    """计算牛顿环干涉图样"""
    ym = 1.5e-3 * scale
    xs = np.linspace(-ym, ym, 500)
    ys = np.linspace(-ym, ym, 500)
    X, Y = np.meshgrid(xs, ys)

    R = st.session_state.R_mm * 1e-3
    shift_x = st.session_state.get('total_shift', 0.0)
    r = np.sqrt((X + shift_x) ** 2 + Y ** 2)

    dd = r ** 2 / R
    I = 1 + np.cos(2 * np.pi * dd / LAMBDA_M)
    B = (I / 4.0) * 255
    return B


def plot_newtons_rings(scale: float = 1.0) -> plt.Figure:
    """绘制牛顿环原理图和干涉图样"""
    fig = plt.figure(figsize=(13, 6), dpi=100)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.2])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # 原理图
    rectangle = patches.Rectangle((-20, -22), 40, 4, linewidth=1, edgecolor='black', facecolor='none')
    ax1.add_patch(rectangle)

    if st.session_state.R_mm == 1500:
        arc = patches.Arc((0, -5), 40, 26, theta1=-180, theta2=0, linewidth=1, edgecolor='black', facecolor='none')
        ax1.add_patch(arc)
        ax1.plot([-20, 20], [-5, -5], linestyle='-', color='k', linewidth=1)
        ax1.plot([0, 0], [10, -18], linestyle='--', color='k', linewidth=1)
        ax1.plot([-14, 0], [-14, -14], linestyle='--', color='k', linewidth=1)
        ax1.plot([-14, -14], [-14, -18], linestyle='-', color='r', linewidth=1)
        ax1.plot([0, -14], [10, -14], linestyle='-', color='r', linewidth=1)
        ax1.text(1, 11, "O", fontsize=14, color='g')
        ax1.text(-11, -2, "R", fontsize=14, color='g')
        ax1.text(-6, -13, "r", fontsize=14, color='g')
        ax1.text(-20, -17, "e", fontsize=14, color='r')
    else:
        arc = patches.Arc((0, -8), 32, 20, theta1=-180, theta2=0, linewidth=1, edgecolor='black', facecolor='none')
        ax1.add_patch(arc)
        ax1.plot([-16, 16], [-8, -8], linestyle='-', color='k', linewidth=1)
        ax1.plot([0, 0], [8, -18], linestyle='--', color='k', linewidth=1)
        ax1.plot([-11, 0], [-15.2, -15.2], linestyle='--', color='k', linewidth=1)
        ax1.plot([-11, -11], [-15.2, -18], linestyle='-', color='r', linewidth=1)
        ax1.plot([0, -11], [8, -15.2], linestyle='-', color='r', linewidth=1)
        ax1.text(1, 11, "O", fontsize=14, color='g')
        ax1.text(-5, -5, "R", fontsize=14, color='g')
        ax1.text(-5, -14, "r", fontsize=14, color='g')
        ax1.text(-15, -17, "e", fontsize=14, color='r')

    ax1.set_ylim(-28, 28)
    ax1.set_xlim(-28, 28)
    ax1.set_facecolor('lightgray')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('牛顿环原理图')

    # 干涉图样
    B = calculate_newtons_rings(scale)
    ax2.imshow(B, cmap='YlOrBr', origin='lower', extent=[-10, 10, -10, 10])
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.set_title(f"牛顿环干涉图样（缩放 ×{scale:.2f}）")
    ax2.grid(alpha=0.3)
    ax2.plot([-1.5, 1.5], [0, 0], color='black', linewidth=1.0)
    ax2.plot([0, 0], [-1.5, 1.5], color='black', linewidth=1.0)

    plt.tight_layout(pad=2)
    return fig


def add_watermark(img: Image.Image, R_mm: float) -> Image.Image:
    """添加水印"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    line1 = f"实验日期：{now}"
    line2 = f"牛顿环半径={R_mm:.2f}mm"
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("SourceHanSansCN-Regular1.otf", 26)
    except Exception:
        font = ImageFont.load_default()
    draw.text((20, img.height - 65), line1, fill=(255, 0, 0), font=font)
    draw.text((20, img.height - 30), line2, fill=(255, 0, 0), font=font)
    return img


@st.dialog("曲率半径不确定度计算结果", width="medium")
def show_uncertainty_dialog() -> None:
    """显示不确定度计算结果"""
    res = calculate_uncertainty()
    if not res:
        st.error("❌ 数据不完整，请先完成左右全部数据采集！")
        return

    left_data = st.session_state.left_pos
    right_data = st.session_state.right_pos

    left_row = [f"{left_data[n]:.3f}" if left_data[n] is not None else "未采集" for n in RECORD_N]
    right_row = [f"{right_data[n]:.3f}" if right_data[n] is not None else "未采集" for n in RECORD_N]

    df = pd.DataFrame([left_row, right_row], columns=RECORD_N, index=["左/mm", "右/mm"])
    df.columns.name = "环数/个"
    st.dataframe(df, use_container_width=True)

    st.markdown(f"""
        <div style="font-size:18px; line-height:1.8;">
        <h3 style='text-align:center;'>✅ 实验数据处理结果（逐差法）</h3>
        <hr>
        <b>入射光波长：</b> 黄光 {res['lambda_true']} nm<br> 
        <b>透镜理论半径：</b> {res['R_theory']} mm<br>
        <b>测量半径 R：</b> {res['R_measured']} mm<br>
        <b>相对误差：</b> {res['rel_error']} %<br>
        <hr>
        <b>A类不确定度 u_A：</b> ±{res['u_A']} mm<br>
        <b>B类不确定度 u_B：</b> ±{res['u_B']} mm<br>
        <b>合成不确定度 u_C：</b> ±{res['u_C']} mm<br>
        <hr>
        <p style='color:green;'>数据有效，可直接用于实验报告</p>
        </div>
    """, unsafe_allow_html=True)


@st.dialog("实验原理", width="small")
def show_principle_dialog() -> None:
    st.markdown("""
        <div style="font-size:17px;"> 
        1. 牛顿环为等厚干涉现象<br>
        2. 光程差公式：Δ = 2d + λ/2<br>
        3. 明纹条件：2d + λ/2 = Kλ<br>
        4. 暗纹条件：2d + λ/2 = (2K+1)λ/2<br>
        5. 实验波长：黄光 589.3nm<br>
        </div> 
    """, unsafe_allow_html=True)


@st.dialog("使用说明", width="small")
def show_guide_dialog() -> None:
    st.markdown("""
        <div style="font-size:17px; line-height:1.6;">
        <b>📋 使用步骤</b><br>
        1. 切换演示/实验模式<br>
        2. 调节牛顿环半径与缩放<br>
        3. 移动条纹并采集实验数据<br>
        4. 计算不确定度<br>
        </div>
    """, unsafe_allow_html=True)


def get_step_to_adjacent_fringe(direction: str = 'right') -> float:
    """计算移动到相邻条纹的步长"""
    R = st.session_state.R_mm * 1e-3
    lam = 589.3e-9
    current_x = st.session_state.total_shift
    current_abs_x = abs(current_x)

    if current_abs_x == 0:
        current_k = 0
    else:
        current_k = (current_abs_x ** 2) / (lam * R)

    if direction == 'right':
        if current_x >= 0:
            r_current = np.sqrt(current_k * lam * R) if current_k > 0 else 0
            r_next = np.sqrt((current_k + 2) * lam * R)
            step = r_next - r_current
        else:
            if current_k <= 2:
                step = current_abs_x
            else:
                r_current = np.sqrt(current_k * lam * R)
                r_prev = np.sqrt((current_k - 2) * lam * R)
                step = r_current - r_prev
    else:
        if current_x <= 0:
            r_current = np.sqrt(current_k * lam * R) if current_k > 0 else 0
            r_next = np.sqrt((current_k + 2) * lam * R)
            step = r_next - r_current
        else:
            if current_k <= 2:
                step = current_abs_x
            else:
                r_current = np.sqrt(current_k * lam * R)
                r_prev = np.sqrt((current_k - 2) * lam * R)
                step = r_current - r_prev
    return step


def main() -> None:
    st.set_page_config(page_title="牛顿环干涉实验", page_icon="🔬", layout="wide")

    # 检测移动端
    mobile_view = False
    try:
        headers = st.context.headers
        user_agent = headers.get('User-Agent', '').lower()
        mobile_view = 'mobile' in user_agent or 'android' in user_agent or 'iphone' in user_agent
    except Exception:
        mobile_view = False

    st.markdown(
        """ 
        <style> 
        /* 隐藏顶部标题栏 + 右上角三个点菜单 */
        header[data-testid="stHeader"] { display: none !important; }
        /* 隐藏右下角的两个图标（帮助+升级） */
        div[data-testid="stToolbar"] { display: none !important; }
        .stApp > div:nth-child(3) { display: none !important; }

        html, body, [class*="stText"] { font-size: 16px !important; } 
        .stButton>button { font-size: 15px !important; } 
        .stNumberInput, .stSelectbox { font-size: 15px !important; } 

        /* 手机竖屏样式 */
        @media only screen and (max-width: 768px) and (orientation: portrait) {
            .stMarkdown h3 {
                display: none !important;
            }
            .stMarkdown h4 {
                font-size: 15px !important;
                font-weight: normal !important;
                margin-top: 4px !important;
                margin-bottom: 4px !important;
            }
        }

        @media only screen and (max-width: 1024px) {
            .block-container { padding: 0px 10px 10px 10px !important; max-width: 100% !important; }
            img, .stPyplot { max-height: 65vh !important; object-fit: contain !important; }
        }
        </style> 
        """,
        unsafe_allow_html=True,
    )

    # 移动端按钮
    if mobile_view:
        st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)
        bt1, bt2 = st.columns(2)
        with bt1:
            if st.button("实验原理", use_container_width=True):
                show_principle_dialog()
        with bt2:
            if st.button("使用说明", use_container_width=True):
                show_guide_dialog()
        st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 1])

    # 初始化session state
    if "initialized" not in st.session_state:
        st.session_state.mode = "demo"
        st.session_state.h = DEFAULT_H
        st.session_state.fringe_count = 0
        st.session_state.left_pos = {n: None for n in RECORD_N}
        st.session_state.right_pos = {n: None for n in RIGHT_RECORD_N}
        st.session_state.drift_seed = random.uniform(-1, 1)
        st.session_state.drift_direction = random.choice([-1, 1])
        st.session_state.zoom_scale = ZOOM_DEFAULT
        st.session_state.fringe_phase = DEFAULT_FRINGE_PHASE
        st.session_state.total_shift = 0.0
        st.session_state.usage_records_demo_export = []
        st.session_state.usage_records_sim_complete = []
        st.session_state.reset_counter = 0
        st.session_state.experiment_completed = False
        st.session_state.initialized = True
        st.session_state.R_mm = 800

    if "need_reset" not in st.session_state:
        st.session_state.need_reset = False

    with col_right:
        if not mobile_view:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("实验原理", use_container_width=True):
                    show_principle_dialog()
            with c2:
                if st.button("使用说明", use_container_width=True):
                    show_guide_dialog()

        if st.button("重置", use_container_width=True):
            reset_all_to_default()
            st.rerun()

        col_demo, col_mode = st.columns(2)
        with col_demo:
            btn_type = "primary" if st.session_state.mode == "demo" else "secondary"
            if st.button("演示模式", use_container_width=True, type=btn_type):
                st.session_state.mode = "demo"
                reset_all_to_default()
                st.rerun()

        with col_mode:
            btn_type = "primary" if st.session_state.mode == "sim" else "secondary"
            if st.button("实验模式", use_container_width=True, type=btn_type):
                st.session_state.mode = "sim"
                reset_all_to_default()
                st.rerun()

        demo_enabled = st.session_state.mode == "demo"
        sim_enabled = st.session_state.mode == "sim"

        # 半径选择
        col_lbl, col_ctrl = st.columns([1, 2])
        with col_lbl:
            st.markdown("牛顿环半径")
        with col_ctrl:
            R_options = [800, 1500]
            selected_R = st.selectbox(
                "牛顿环半径",
                options=R_options,
                index=R_options.index(st.session_state.R_mm),
                label_visibility="collapsed",
                key=f"R_select_{st.session_state.reset_counter}",
                format_func=lambda x: f"{x} mm"
            )
            if selected_R != st.session_state.R_mm:
                st.session_state.R_mm = selected_R
                st.rerun()

        col_left_move, col_right_move = st.columns(2)

        if demo_enabled:
            with col_left_move:
                if st.button("向左移动", use_container_width=True, key="left_move_btn"):
                    step = get_step_to_adjacent_fringe(direction='left')
                    if step > 0:
                        st.session_state.total_shift -= step
                        st.session_state.zoom_scale = min(st.session_state.zoom_scale + 0.05, ZOOM_MAX)
                        st.rerun()

            with col_right_move:
                if st.button("向右移动", use_container_width=True, key="right_move_btn"):
                    step = get_step_to_adjacent_fringe(direction='right')
                    if step > 0:
                        st.session_state.total_shift += step
                        st.session_state.zoom_scale = min(st.session_state.zoom_scale + 0.05, ZOOM_MAX)
                        st.rerun()

        else:  # 实验模式
            with col_left_move:
                if st.button("左34级条纹", use_container_width=True, key="left_34_btn"):
                    R = st.session_state.R_mm * 1e-3
                    lam = 589.3e-9
                    st.session_state.total_shift = -np.sqrt(34 * lam * R)
                    st.rerun()

            with col_right_move:
                if st.button("向右移动2级", use_container_width=True, key="right_move_btn_sim"):
                    R = st.session_state.R_mm * 1e-3
                    lam = 589.3e-9

                    step = get_step_to_adjacent_fringe(direction='right')
                    if step > 0:
                        st.session_state.total_shift += step

                    current_x = st.session_state.total_shift
                    current_abs_x = abs(current_x)
                    current_k = int(round((current_abs_x ** 2) / (lam * R)))
                    current_k = min(current_k, 32)
                    if current_k % 2 != 0:
                        current_k -= 1

                    # 左侧数据生成
                    if current_x < 0 and current_k in RECORD_N:
                        if st.session_state.left_pos[current_k] is None:
                            r_mm = np.sqrt(current_k * LAMBDA_MM * st.session_state.R_mm)
                            noisy_val = add_measurement_error(-round(r_mm, 3))
                            st.session_state.left_pos[current_k] = noisy_val
                            st.success(f"✅ 已采集【左侧】第 {current_k} 级暗纹数据：{noisy_val:.3f} mm")

                    # 右侧数据生成
                    if current_x > 0 and current_k in RIGHT_RECORD_N:
                        if st.session_state.right_pos[current_k] is None:
                            r_mm = np.sqrt(current_k * LAMBDA_MM * st.session_state.R_mm)
                            noisy_val = add_measurement_error(round(r_mm, 3))
                            st.session_state.right_pos[current_k] = noisy_val
                            st.success(f"✅ 已采集【右侧】第 {current_k} 级暗纹数据：{noisy_val:.3f} mm")

                    st.session_state.zoom_scale = min(st.session_state.zoom_scale + 0.05, ZOOM_MAX)
                    st.rerun()

        # 演示模式导出
        if demo_enabled:
            st.markdown("演示结果导出")
            demo_export_remaining = get_demo_export_remaining()
            export_disabled = demo_export_remaining <= 0

            col_export, col_download = st.columns(2)
            with col_export:
                btn_text = f"导出PNG ({demo_export_remaining}次)" if demo_export_remaining > 0 else "导出PNG (已用完)"
                if st.button(btn_text, width="stretch", disabled=export_disabled):
                    if check_demo_export_quota():
                        consume_demo_export_quota()
                        fig_temp = plot_newtons_rings(st.session_state.zoom_scale)
                        buf = io.BytesIO()
                        fig_temp.savefig(buf, format="png", bbox_inches="tight", dpi=130)
                        buf.seek(0)
                        img = Image.open(buf).convert("RGB")
                        img = add_watermark(img, st.session_state.R_mm)
                        o = io.BytesIO()
                        img.save(o, format="PNG")
                        plt.close(fig_temp)
                        st.session_state.png_data = o.getvalue()
                        st.success("✅ PNG已生成")
                        st.rerun()

            with col_download:
                if "png_data" in st.session_state and demo_enabled:
                    st.download_button("⬇️ 下载PNG", st.session_state.png_data, "牛顿环.png", "image/png", width="stretch")

        # 实验模式数据预览
        if sim_enabled:
            st.markdown("实验数据预览")
            left_data = [st.session_state.left_pos.get(k) for k in RECORD_N]
            left_row = [f"{v:.3f}" if v is not None else "" for v in left_data]
            right_data = [st.session_state.right_pos.get(k) for k in RECORD_N]
            right_row = [f"{v:.3f}" if v is not None else "" for v in right_data]

            df = pd.DataFrame([left_row, right_row], columns=RECORD_N, index=["左/mm", "右/mm"])
            st.dataframe(df, use_container_width=True)

            st.markdown("<br>")
            data_complete = all(v is not None for v in st.session_state.left_pos.values()) and \
                           all(v is not None for v in st.session_state.right_pos.values())

            if data_complete:
                st.button("📊 不确定度", type="primary", use_container_width=True, on_click=show_uncertainty_dialog)
            else:
                remaining = sum(1 for v in st.session_state.left_pos.values() if v is None) + \
                           sum(1 for v in st.session_state.right_pos.values() if v is None)
                st.info(f"📝 还需采集 {remaining} 个数据点")
                st.button("📊 不确定度", disabled=True, use_container_width=True)

    # 左侧显示牛顿环图
    with col_left:
        fig_final = plot_newtons_rings(st.session_state.zoom_scale)
        st.pyplot(fig_final, use_container_width=True)
        plt.close(fig_final)

    if st.session_state.need_reset:
        st.session_state.need_reset = False


if __name__ == "__main__":
    main()
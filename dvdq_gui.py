# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║         Battery dV/dQ Analyzer  —  GUI Edition              ║
║         Anaconda + PyQt5 + matplotlib                        ║
║                                                              ║
║  실행 전 설치:                                               ║
║    conda install pyqt matplotlib scipy pandas numpy          ║
║  또는:                                                       ║
║    pip install PyQt5 matplotlib scipy pandas numpy           ║
║                                                              ║
║  실행:  python dvdq_gui_v16.py                               ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
import os
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d

# ── matplotlib을 Qt 백엔드로 설정 (PyQt5보다 먼저) ──────────
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QPushButton, QSlider, QCheckBox, QScrollArea,
    QFrame, QFileDialog, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QStatusBar, QToolBar, QAction, QMessageBox,
    QComboBox, QButtonGroup, QRadioButton, QSizePolicy, QSpacerItem,
    QListWidget, QListWidgetItem, QAbstractItemView,
    QMenu, QInputDialog, QDialog, QFormLayout, QDialogButtonBox,
    QSplashScreen
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import (
    QFont, QColor, QPalette, QIcon, QPixmap, QPainter,
    QLinearGradient, QBrush, QPen, QPainterPath
)


# ══════════════════════════════════════════════════════════════
#  색상 & 스타일 상수
# ══════════════════════════════════════════════════════════════
BG_DARK       = "#0F0F1A"
BG_PANEL      = "#16162A"
BG_CARD       = "#1E1E35"
BG_HOVER      = "#252545"
ACCENT_BLUE   = "#4CC9F0"
ACCENT_PURPLE = "#7B2FBE"
ACCENT_GREEN  = "#06D6A0"
ACCENT_RED    = "#EF233C"
TEXT_PRIMARY  = "#E8E8F0"
TEXT_SECONDARY= "#8888AA"
BORDER_COLOR  = "#2A2A4A"

CYCLE_PALETTE = [
    ("#E63946", "#4361EE"),   # red  / blue
    ("#F4A261", "#4CC9F0"),   # orange / cyan
    ("#2A9D8F", "#7209B7"),   # teal / purple
    ("#E9C46A", "#3A0CA3"),   # yellow / indigo
    ("#06D6A0", "#EF233C"),   # green / red
    ("#B5179E", "#48CAE4"),   # magenta / sky
    ("#FFB703", "#023E8A"),   # amber / navy
    ("#8338EC", "#FB5607"),   # violet / orange
]

# ══════════════════════════════════════════════════════════════
#  해상도 스케일 팩터
#  기준 해상도: 1920×1080  →  현재 모니터 비율로 자동 스케일
# ══════════════════════════════════════════════════════════════
_BASE_W, _BASE_H = 1920, 1080   # 기준 해상도 (현재 UI가 최적화된 해상도)
UI_SCALE = 1.0                  # main()에서 실제 값으로 교체됨

def px(value: float) -> int:
    """기준 해상도의 px 값을 현재 해상도에 맞게 스케일한 정수 반환."""
    return max(1, round(value * UI_SCALE))

def _scaled_stylesheet() -> str:
    """UI_SCALE이 설정된 뒤 호출하여 스케일된 스타일시트를 반환."""
    return f"""
QMainWindow, QWidget {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: {px(11)}px;
}}
QSplitter::handle {{
    background: {BORDER_COLOR};
    width: {px(2)}px; height: {px(2)}px;
}}
/* ── 패널 ── */
QGroupBox {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER_COLOR};
    border-radius: {px(8)}px;
    margin-top: {px(14)}px;
    padding: {px(8)}px;
    font-weight: 600;
    color: {ACCENT_BLUE};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: {px(10)}px; top: -2px;
    padding: 0 {px(6)}px;
    color: {ACCENT_BLUE};
    font-size: {px(11)}px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}}
/* ── 버튼 ── */
QPushButton {{
    background-color: {BG_CARD};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
    border-radius: {px(6)}px;
    padding: {px(6)}px {px(14)}px;
    font-weight: 600;
    min-height: {px(28)}px;
}}
QPushButton:hover {{
    background-color: {BG_HOVER};
    border-color: {ACCENT_BLUE};
    color: {ACCENT_BLUE};
}}
QPushButton:pressed {{
    background-color: {ACCENT_PURPLE};
    border-color: {ACCENT_PURPLE};
    color: white;
}}
QPushButton#btn_load {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {ACCENT_PURPLE}, stop:1 #4361EE);
    border: none;
    color: white;
    font-size: {px(12)}px;
    font-weight: 700;
    letter-spacing: 1px;
    border-radius: {px(8)}px;
    padding: {px(10)}px {px(18)}px;
}}
QPushButton#btn_load:hover {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #9B3FDE, stop:1 {ACCENT_BLUE});
}}
QPushButton#btn_export {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #0A6E5E, stop:1 {ACCENT_GREEN});
    border: none; color: white; font-weight: 700;
    border-radius: {px(6)}px; padding: {px(7)}px {px(14)}px;
}}
QPushButton#btn_export:hover {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {ACCENT_GREEN}, stop:1 #06D6A0);
}}
/* ── 슬라이더 ── */
QSlider::groove:horizontal {{
    background: {BG_CARD};
    height: {px(4)}px; border-radius: {px(2)}px;
    border: 1px solid {BORDER_COLOR};
}}
QSlider::handle:horizontal {{
    background: {ACCENT_BLUE};
    width: {px(14)}px; height: {px(14)}px;
    border-radius: {px(7)}px;
    margin: -{px(5)}px 0;
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {ACCENT_PURPLE}, stop:1 {ACCENT_BLUE});
    border-radius: {px(2)}px;
}}
/* ── SpinBox ── */
QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_COLOR};
    border-radius: {px(4)}px;
    color: {TEXT_PRIMARY};
    padding: {px(3)}px {px(6)}px;
    min-height: {px(22)}px;
}}
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT_BLUE};
}}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{
    background: {BG_CARD}; color: {TEXT_PRIMARY};
    selection-background-color: {ACCENT_PURPLE};
}}
/* ── 체크박스 ── */
QCheckBox {{
    spacing: {px(6)}px;
    color: {TEXT_PRIMARY};
    font-size: {px(11)}px;
}}
QCheckBox::indicator {{
    width: {px(14)}px; height: {px(14)}px;
    border-radius: {px(3)}px;
    border: 1px solid {BORDER_COLOR};
    background: {BG_CARD};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT_BLUE};
    border-color: {ACCENT_BLUE};
    image: none;
}}
/* ── 스크롤 ── */
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: {BG_DARK}; width: {px(6)}px; border-radius: {px(3)}px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_COLOR}; border-radius: {px(3)}px; min-height: {px(20)}px;
}}
QScrollBar::handle:vertical:hover {{ background: {ACCENT_BLUE}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
/* ── 탭 ── */
QTabWidget::pane {{
    border: 1px solid {BORDER_COLOR}; border-radius: {px(6)}px;
    background: {BG_PANEL};
}}
QTabBar::tab {{
    background: {BG_DARK}; color: {TEXT_SECONDARY};
    padding: {px(6)}px {px(16)}px; border-radius: {px(4)}px {px(4)}px 0 0;
    font-weight: 600; font-size: {px(11)}px;
    border: 1px solid {BORDER_COLOR}; border-bottom: none;
    margin-right: {px(2)}px;
}}
QTabBar::tab:selected {{
    background: {BG_PANEL}; color: {ACCENT_BLUE};
    border-color: {ACCENT_BLUE};
}}
QTabBar::tab:hover:!selected {{
    background: {BG_HOVER}; color: {TEXT_PRIMARY};
}}
/* ── 테이블 ── */
QTableWidget {{
    background: {BG_PANEL}; color: {TEXT_PRIMARY};
    gridline-color: {BORDER_COLOR}; border: none;
    font-size: {px(10)}px;
}}
QTableWidget::item {{ padding: {px(4)}px {px(8)}px; }}
QTableWidget::item:selected {{
    background: {ACCENT_PURPLE}; color: white;
}}
QHeaderView::section {{
    background: {BG_CARD}; color: {ACCENT_BLUE};
    border: none; border-right: 1px solid {BORDER_COLOR};
    padding: {px(5)}px {px(8)}px; font-weight: 700; font-size: {px(10)}px;
    letter-spacing: 0.5px;
}}
/* ── 상태바 ── */
QStatusBar {{
    background: {BG_PANEL}; color: #C8C8E0;
    border-top: 1px solid {BORDER_COLOR};
    font-size: {px(10)}px;
}}
QProgressBar {{
    background: {BG_CARD}; border: 1px solid {BORDER_COLOR};
    border-radius: {px(4)}px; height: {px(6)}px; text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {ACCENT_PURPLE}, stop:1 {ACCENT_BLUE});
    border-radius: {px(4)}px;
}}
/* ── 구분선 ── */
QFrame[frameShape="4"], QFrame[frameShape="5"] {{
    color: {BORDER_COLOR};
}}
/* ── Label ── */
QLabel#lbl_title {{
    font-size: {px(18)}px; font-weight: 800;
    color: {ACCENT_BLUE}; letter-spacing: 2px;
}}
QLabel#lbl_subtitle {{
    font-size: {px(10)}px; color: {TEXT_SECONDARY};
    letter-spacing: 1px;
}}
QLabel#lbl_stat {{
    background: {BG_CARD}; border: 1px solid {BORDER_COLOR};
    border-radius: {px(6)}px; padding: {px(6)}px {px(10)}px;
    color: {TEXT_PRIMARY}; font-size: {px(11)}px;
}}
QMenu {{
    background: {BG_CARD}; color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR}; border-radius: {px(6)}px; padding: {px(4)}px;
}}
QMenu::item {{ padding: {px(6)}px {px(20)}px; border-radius: {px(4)}px; }}
QMenu::item:selected {{ background: {ACCENT_PURPLE}; color: white; }}
"""


# ══════════════════════════════════════════════════════════════
#  분석 코어 (이전 단계 클래스 포함)
# ══════════════════════════════════════════════════════════════
COL_MAP = {
    'time':     ['time/s','time','t','시간'],
    'voltage':  ['ewe/v','ecell/v','voltage','v','volt','전압','u/v'],
    'current':  ['i/ma','current','i','전류','i/a','<i>/ma'],
    'capacity': ['q/mah','q charge/discharge/mah','capacity','cap',
                 '(q-qo)/mah','q charge/discharge/ma.h','q','용량'],
    'cycle':    ['cycle number','cycle','cyc','사이클','ns'],
    'half_cyc': ['half cycle','half_cycle','ox/red'],
}

def _detect_mpt(filepath):
    """EC-Lab MPT ASCII 파일 여부 판별 및 헤더 줄 수 반환."""
    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            first = f.readline()
            if 'EC-Lab ASCII FILE' not in first:
                return None
            for line in f:
                if line.startswith('Nb header lines'):
                    return int(line.split(':')[1].strip())
    except Exception:
        pass
    return None

def _load_mpt(filepath, n_header):
    """MPT 파일 → 표준 DataFrame 변환."""
    df = pd.read_csv(filepath, sep='\t', skiprows=n_header - 1,
                     encoding='latin-1', low_memory=False)

    # 숫자 변환
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # 컬럼 표준화
    col_lower = {c.strip().lower(): c for c in df.columns}
    rename = {}
    for std, cands in COL_MAP.items():
        for cand in cands:
            if cand in col_lower:
                rename[col_lower[cand]] = std
                break

    df = df.rename(columns=rename)

    # MPT 전용: Q charge/discharge/mA.h → capacity 보완
    if 'capacity' not in df.columns:
        for raw_name in df.columns:
            if 'q charge/discharge' in raw_name.lower():
                df['capacity'] = df[raw_name].abs()
                break

    # MPT 전용: cycle number 0-base → 1-base 변환
    if 'cycle' in df.columns:
        df['cycle'] = df['cycle'].fillna(0)
        min_cyc = df['cycle'].min()
        if min_cyc == 0:
            # cycle 0은 formation/pre-cycle → 1부터 시작하도록 +1
            df['cycle'] = (df['cycle'] + 1).astype(int)

    # MPT 전용: ox/red 컬럼으로 충방전 방향 보완
    if 'current' not in df.columns and 'ox/red' in df.columns:
        df['current'] = np.where(df['ox/red'] == 1, 1.0, -1.0)

    # capacity 절댓값 (방전 시 음수인 경우 대응)
    if 'capacity' in df.columns:
        df['capacity'] = df['capacity'].abs()

    # 유효 행만 남기기
    df = df.dropna(subset=['voltage', 'capacity']).reset_index(drop=True)

    # 전압 범위 필터 (OCV 노이즈 제거: |V| < 0.5V 이고 I≈0 구간 제외)
    if 'current' in df.columns:
        is_rest = (df['current'].abs() < 1e-4) & (df['voltage'].abs() < 0.5)
        df = df[~is_rest].reset_index(drop=True)

    return df

def _load_csv(filepath):
    """CSV/TSV/TXT 범용 로더."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        line = f.readline()
    sep = '\t' if '\t' in line else (';' if ';' in line else ',')
    df = pd.read_csv(filepath, sep=sep, engine='python', skip_blank_lines=True)

    rename = {}
    for std, cands in COL_MAP.items():
        for col in df.columns:
            if col.strip().lower() in cands:
                rename[col] = std; break
    df = df.rename(columns=rename)

    for c in ['voltage','current','capacity','time','cycle','half_cyc']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(subset=['voltage','capacity']).reset_index(drop=True)

    if 'cycle' not in df.columns:
        df['cycle'] = 1
    if 'current' not in df.columns:
        dv = df['voltage'].diff()
        df['current'] = np.where(dv >= 0, 1.0, -1.0)

    return df


def _parse_cycle_number(sheet_name):
    """시트명에서 사이클 번호 추출.
    예: 'Cap vs. V 3rd cycle' → 3,  'Cap vs. V 53rd cycle' → 53"""
    import re
    m = re.search(r'(\d+)', sheet_name)
    return int(m.group(1)) if m else None


def _load_excel(filepath):
    """EC-Lab 스타일 Excel 로더.

    시트 구조 (시트 하나 = 사이클 하나):
      헤더 행:  Capacity/mA.h/g | Ecell/V | Q charge/mA.h/g | Ecell/V |
                Q discharge/mA.h/g | Ecell/V
      이후 행:  데이터 (None = 해당 구간 없음)

    반환: 표준 DataFrame (cycle, capacity, voltage, current 컬럼 포함)
    """
    import openpyxl
    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)

    records = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        cyc_no = _parse_cycle_number(sheet_name)
        if cyc_no is None:
            continue   # 'Cell info' 등 비-사이클 시트 스킵

        rows = list(ws.iter_rows(values_only=True))
        if len(rows) < 2:
            continue

        header = [str(h).strip() if h is not None else '' for h in rows[0]]

        # 컬럼 인덱스 자동 탐색 (헤더 키워드 기반)
        def _find_col(keywords):
            for ki, kw in enumerate(keywords):
                for ci, h in enumerate(header):
                    if kw.lower() in h.lower():
                        return ci
            return None

        # charge: Q charge + Ecell (charge)
        ci_qchg = _find_col(['q charge', 'charge'])
        ci_vchg = ci_qchg + 1 if ci_qchg is not None else None

        # discharge: Q discharge + Ecell (discharge)
        ci_qdis = _find_col(['q discharge', 'discharge'])
        ci_vdis = ci_qdis + 1 if ci_qdis is not None else None

        # fallback: full capacity + ecell (col 0, 1)
        ci_qfull = 0 if len(header) >= 2 else None
        ci_vfull = 1 if len(header) >= 2 else None

        # ── 충전 데이터 추출 ──────────────────────────────────
        if ci_qchg is not None and ci_vchg is not None:
            for row in rows[1:]:
                q = row[ci_qchg]
                v = row[ci_vchg]
                if q is None or v is None:
                    continue
                try:
                    records.append({
                        'cycle':    cyc_no,
                        'capacity': float(q),
                        'voltage':  float(v),
                        'current':  1.0,      # 충전 = 양수
                    })
                except (TypeError, ValueError):
                    continue

        # ── 방전 데이터 추출 ──────────────────────────────────
        if ci_qdis is not None and ci_vdis is not None:
            for row in rows[1:]:
                q = row[ci_qdis]
                v = row[ci_vdis]
                if q is None or v is None:
                    continue
                try:
                    records.append({
                        'cycle':    cyc_no,
                        'capacity': float(q),
                        'voltage':  float(v),
                        'current':  -1.0,     # 방전 = 음수
                    })
                except (TypeError, ValueError):
                    continue

        # ── fallback: charge/discharge 컬럼 없으면 full 컬럼 사용 ──
        if ci_qchg is None and ci_qdis is None and ci_qfull is not None:
            for row in rows[1:]:
                q = row[ci_qfull]
                v = row[ci_vfull]
                if q is None or v is None:
                    continue
                try:
                    records.append({
                        'cycle':    cyc_no,
                        'capacity': float(q),
                        'voltage':  float(v),
                        'current':  1.0,
                    })
                except (TypeError, ValueError):
                    continue

    wb.close()

    if not records:
        raise ValueError("Excel 파일에서 유효한 데이터를 찾을 수 없습니다.\n"
                         "시트명 패턴: 'Cap vs. V {N}th cycle' 또는 사이클 번호 포함 필요.")

    df = pd.DataFrame(records)
    df = df.dropna(subset=['voltage', 'capacity']).reset_index(drop=True)
    return df


def load_data(filepath):
    """MPT / CSV / TXT / XLSX 자동 판별 후 표준 DataFrame 반환."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext in ('.xlsx', '.xls'):
        df = _load_excel(filepath)
        df._file_type = 'xlsx'
    else:
        n_header = _detect_mpt(filepath)
        if n_header is not None:
            df = _load_mpt(filepath, n_header)
            df._file_type = 'mpt'
        else:
            df = _load_csv(filepath)
            df._file_type = 'csv'

    if 'cycle' not in df.columns:
        df['cycle'] = 1
    df['cycle'] = pd.to_numeric(df['cycle'], errors='coerce').fillna(1).astype(int)

    return df

def _remove_cv_rest(seg_df, direction='charge'):
    """CC 구간만 남기고 CV 및 Rest 구간 제거.

    충전(charge): 말단 CV 테일 제거
                  CC 전류 중앙값의 70% 미만으로 떨어지는 말단 구간을 잘라냄
    방전(discharge): 양끝 Rest/CV 제거
    """
    if 'current' not in seg_df.columns or len(seg_df) < 10:
        return seg_df

    I    = seg_df['current'].values.astype(float)
    absI = np.abs(I)

    # Rest 제거 (공통): |I| < 전체 median 의 5%
    pos_mask = absI > 0
    if not pos_mask.any():
        return seg_df
    global_med = np.median(absI[pos_mask])
    rest_mask  = absI > global_med * 0.05
    seg_df     = seg_df[rest_mask].reset_index(drop=True)
    if len(seg_df) < 5:
        return seg_df
    I    = seg_df['current'].values.astype(float)
    absI = np.abs(I)

    if direction == 'charge':
        # 충전 CC 말단 CV 제거
        cc_median = np.median(absI)
        cv_thr    = cc_median * 0.70
        cut_idx   = len(absI)
        for i in range(len(absI) - 1, -1, -1):
            if absI[i] >= cv_thr:
                cut_idx = i + 1
                break
        seg_df = seg_df.iloc[:cut_idx].reset_index(drop=True)
    else:
        # 방전: 양끝 CV/Rest 제거
        cc_median = np.median(absI)
        cv_thr    = cc_median * 0.30
        cc_mask   = absI >= cv_thr
        result    = seg_df[cc_mask].reset_index(drop=True)
        seg_df    = result if len(result) > 5 else seg_df

    return seg_df


def split_cycles(df):
    """사이클별 충전/방전 세그먼트 분리.

    MPT 파일: half cycle 컬럼이 있으면 half cycle 기준으로 분리
              → 같은 cycle 내에서도 CC/CV 복수 구간을 하나로 합침
    CSV 파일: current 부호 기준으로 분리
    """
    segs = {}

    has_half = ('half_cyc' in df.columns and
                df['half_cyc'].notna().any() and
                df['half_cyc'].nunique() > 1)

    for cyc, cdf in df.groupby('cycle', sort=True):
        cyc = int(cyc)

        if has_half:
            # half cycle 짝수 = 충전, 홀수 = 방전 (Bio-Logic 관례)
            # ox/red=1 → oxidation → 충전
            if 'current' in cdf.columns:
                chg_mask = cdf['current'] >= 0
                dis_mask = cdf['current'] <  0
            else:
                chg_mask = cdf['half_cyc'] % 2 == 0
                dis_mask = cdf['half_cyc'] % 2 == 1

            # 충전: Q 오름차순 (각 half_cycle 내 Q reset 고려)
            chg = cdf[chg_mask].copy()
            dis = cdf[dis_mask].copy()

            # half_cycle별로 Q를 누적 재계산
            def rebuild_Q(sub):
                """half_cycle 단위로 Q를 연속 누적값으로 재구성."""
                parts = []
                offset = 0.0
                for _, grp in sub.groupby('half_cyc', sort=True):
                    q = grp['capacity'].values.copy()
                    q_min = q.min() if len(q) else 0
                    q = q - q_min + offset
                    offset = q.max() if len(q) else offset
                    g2 = grp.copy()
                    g2['capacity'] = q
                    parts.append(g2)
                return pd.concat(parts).reset_index(drop=True) if parts else sub

            if has_half and len(chg) > 5:
                chg = rebuild_Q(chg)
            if has_half and len(dis) > 5:
                dis = rebuild_Q(dis)

            chg = chg.sort_values('capacity')
            dis = dis.sort_values('capacity', ascending=False)
        else:
            chg = cdf[cdf['current'] >= 0].sort_values('capacity')
            dis = cdf[cdf['current'] <  0].sort_values('capacity', ascending=False)

        # CV / Rest 구간 제거 (충전: 말단 CV tail 제거, 방전: 양끝 제거)
        chg = _remove_cv_rest(chg, direction='charge')
        dis = _remove_cv_rest(dis, direction='discharge')

        if len(chg) > 5: segs[(cyc, 'charge')]    = chg.reset_index(drop=True)
        if len(dis) > 5: segs[(cyc, 'discharge')]  = dis.reset_index(drop=True)

    return segs

def compute_dvdq(seg_df, smooth_w=11, interp_n=500,
                 dvdq_clip=50.0, peak_prom=0.3):
    Q = seg_df['capacity'].values.astype(float)
    V = seg_df['voltage'].values.astype(float)
    mask = ~(np.isnan(Q) | np.isnan(V))
    Q, V = Q[mask], V[mask]
    _, idx = np.unique(Q, return_index=True)
    Q, V = Q[idx], V[idx]
    if len(Q) < 5:
        return None

    # ① 보간: 균일 Q 그리드 위에 V(Q) 재구성
    Q_i = np.linspace(Q.min(), Q.max(), interp_n)
    kind = 'linear' if len(Q) < 4 else 'cubic'
    V_i = interp1d(Q, V, kind=kind, fill_value='extrapolate')(Q_i)

    # ② 스무딩: 미분 전에 V 곡선에 Savitzky-Golay 적용
    #    (Dubarry & Anseán 2022 best-practice: smooth V before differentiation)
    w = smooth_w if smooth_w % 2 == 1 else smooth_w + 1
    w = max(w, 5)
    if len(V_i) >= w:
        V_i_s = savgol_filter(V_i, window_length=w, polyorder=min(3, w - 1))
    else:
        V_i_s = V_i.copy()

    # ③ 미분: 스무딩된 V 곡선을 수치 미분 후 clipping
    dvdq   = np.clip(np.gradient(V_i,   Q_i), -dvdq_clip, dvdq_clip)   # 원본 보존(표시용)
    dvdq_s = np.clip(np.gradient(V_i_s, Q_i), -dvdq_clip, dvdq_clip)   # 스무딩 후 미분

    pp, _ = find_peaks( dvdq_s, prominence=peak_prom)
    pn, _ = find_peaks(-dvdq_s, prominence=peak_prom)

    # ── 자동 zoom 범위 계산 (초반/말단 급등락 구간 제외) ──────
    # dV/dQ 의 |값| 이 안정된 중간 구간 추정 (상위 5% trim)
    trim_lo = np.percentile(Q_i, 5)
    trim_hi = np.percentile(Q_i, 95)
    mid_mask = (Q_i >= trim_lo) & (Q_i <= trim_hi)
    # 중간 구간에서 |dV/dQ| 이 clip 값의 80% 미만인 구간으로 좁힘
    stable = mid_mask & (np.abs(dvdq_s) < dvdq_clip * 0.8)
    if stable.sum() > 10:
        Q_zoom_lo = float(Q_i[stable].min())
        Q_zoom_hi = float(Q_i[stable].max())
        V_zoom_lo = float(V_i[stable].min())
        V_zoom_hi = float(V_i[stable].max())
        dv_zoom_lo = float(np.percentile(dvdq_s[stable], 2))
        dv_zoom_hi = float(np.percentile(dvdq_s[stable], 98))
    else:
        Q_zoom_lo, Q_zoom_hi = float(Q_i.min()), float(Q_i.max())
        V_zoom_lo, V_zoom_hi = float(V_i.min()), float(V_i.max())
        dv_zoom_lo = float(np.percentile(dvdq_s, 2))
        dv_zoom_hi = float(np.percentile(dvdq_s, 98))

    return dict(
        Q_raw=Q, V_raw=V, Q_i=Q_i, V_i=V_i,
        dvdq_r=dvdq, dvdq_s=dvdq_s,
        dvdq_pk=dict(pos=(Q_i[pp], dvdq_s[pp]), neg=(Q_i[pn], dvdq_s[pn])),
        Q_total=float(Q.max() - Q.min()),
        V_min=float(V.min()), V_max=float(V.max()),
        zoom=dict(Q_lo=Q_zoom_lo, Q_hi=Q_zoom_hi,
                  V_lo=V_zoom_lo, V_hi=V_zoom_hi,
                  dv_lo=dv_zoom_lo, dv_hi=dv_zoom_hi,
),
    )


# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
#  DVA 분석 코어  —  Dahn et al. JES 2012, Eq.[1]
#
#  dV/dQ(Q) = (1/mp)·dVp/dqp  −  (1/mn)·dVn/dqn
#
#  여기서
#    Q  = qp·mp + δp   (양극 용량 → 전지 용량으로 변환)
#    Q  = qn·mn + δn   (음극 용량 → 전지 용량으로 변환)
#
#  알고리즘:
#   1. 각 반쪽전지 raw data → q(비용량, mAh) 기준 정규화
#   2. 슬리피지(δ) 적용: Q = q·m + δ  으로 전지 용량축 변환
#   3. 공통 Q 그리드에서 (1/m)·dV/dq 보간
#   4. 양극항 − 음극항 → Estimated dV/dQ
# ══════════════════════════════════════════════════════════════
def _seg_normalize_Q(seg_df):
    """세그먼트 raw data → (Q_norm, V) 반환.
    Q_norm: 0에서 시작하는 단조 용량 (최솟값을 0으로 shift)."""
    Q = seg_df['capacity'].values.astype(float)
    V = seg_df['voltage'].values.astype(float)
    mask = ~(np.isnan(Q) | np.isnan(V))
    Q, V = Q[mask], V[mask]
    if len(Q) < 5:
        return None, None
    # Q 단조 증가 보장 (이미 split_cycles에서 정렬되었지만 재확인)
    order = np.argsort(Q)
    Q, V = Q[order], V[order]
    _, uid = np.unique(Q, return_index=True)
    Q, V = Q[uid], V[uid]
    Q = Q - Q.min()          # 0 기준 정규화
    return Q, V


def _dvdq_on_Q_grid(Q_raw, V_raw, mass, slip, Q_grid,
                    interp_n=2000, smooth_w=21, dvdq_clip=50.0):
    """
    반쪽전지 (Q_raw, V_raw) → 전지 Q 그리드 위의 mass·dV/dQ 계산.

    Dahn JES 2012 원래 정의:
      Q_cell = Q_raw / mass + slip
      dV/dQ_cell = mass · dV/dq_half

    처리 순서 (Dubarry & Anseán 2022 best-practice):
      ① 보간 → ② 스무딩(V) → ③ 미분

    반환: Q_grid 위에서 보간된 mass·dV/dq  배열 (길이 = len(Q_grid))
    """
    if Q_raw is None or len(Q_raw) < 5:
        return np.zeros_like(Q_grid)

    # ① 보간: 비용량 그리드에서 V 재구성 (고밀도)
    Q_interp = np.linspace(Q_raw.min(), Q_raw.max(), interp_n)
    kind = 'linear' if len(Q_raw) < 4 else 'cubic'
    V_interp = interp1d(Q_raw, V_raw,
                        kind=kind, fill_value='extrapolate')(Q_interp)

    # ② 스무딩: 미분 전에 V 곡선에 Savitzky-Golay 적용
    w = smooth_w if smooth_w % 2 == 1 else smooth_w + 1
    w = max(w, 5)
    if len(V_interp) >= w:
        V_interp = savgol_filter(V_interp,
                                 window_length=w,
                                 polyorder=min(3, w - 1))

    # ③ 미분: 스무딩된 V에서 dV/dq 계산 후 clipping
    dVdq_half = np.gradient(V_interp, Q_interp)
    dVdq_half = np.clip(dVdq_half, -dvdq_clip / mass, dvdq_clip / mass)

    # ④ 전지 Q 축으로 변환:  Q_cell = Q_raw / mass + slip
    Q_cell = Q_interp / mass + slip   # 전지 용량 축

    # ⑤  mass · dV/dq  =  dV/dQ_cell
    dVdQ_cell = dVdq_half * mass

    # ⑥ Q_grid 위로 보간 (범위 밖은 0)
    f = interp1d(Q_cell, dVdQ_cell,
                 kind='linear', bounds_error=False, fill_value=0.0)
    return f(Q_grid)


def dva_compute_estimated(
        pos_seg, neg_seg,
        pos_mass,      # 양극 활물질 질량 (g 또는 임의 단위, 상대값)
        neg_mass,      # 음극 활물질 질량
        pos_slip,      # 양극 슬리피지 δp (mAh)
        neg_slip,      # 음극 슬리피지 δn (mAh)
        interp_n=2000,
        smooth_w=21,
        dvdq_clip=50.0,
        n_grid=500,
    ):
    """
    Dahn JES 2012 Eq.[1] 구현:

        dV/dQ(Q) = (1/mp)·dVp/dqp(qp)  −  (1/mn)·dVn/dqn(qn)

    where  Q = qp·mp + δp  and  Q = qn·mn + δn

    반환 dict:
        Q_grid        : 공통 전지 Q 축 (mAh)
        dvdq_pos_scaled : (1/mp)·dVp/dqp  at Q_grid
        dvdq_neg_scaled : (1/mn)·dVn/dqn  at Q_grid
        dvdq_estimated  : 양극 − 음극 합산 (스무딩 적용)
        Q_pos_cell    : 양극 Q_cell 범위 (표시용)
        Q_neg_cell    : 음극 Q_cell 범위 (표시용)
    """
    # ── 데이터 전처리 ──────────────────────────────────────────
    Q_pos, V_pos = _seg_normalize_Q(pos_seg)
    Q_neg, V_neg = _seg_normalize_Q(neg_seg)
    if Q_pos is None or Q_neg is None:
        return None

    # ── 전지 Q 축 범위 결정 ────────────────────────────────────
    # Dahn 정의: Q_cell = q / mass + slip
    Q_pos_cell_lo = Q_pos.min() / pos_mass + pos_slip
    Q_pos_cell_hi = Q_pos.max() / pos_mass + pos_slip
    Q_neg_cell_lo = Q_neg.min() / neg_mass + neg_slip
    Q_neg_cell_hi = Q_neg.max() / neg_mass + neg_slip

    # 양극·음극의 합집합(union) 범위 사용
    # 한쪽 전극이 끝난 구간은 _dvdq_on_Q_grid의 fill_value=0.0 으로 자동 처리됨
    # → 한쪽이 끝나도 나머지 전극의 그래프가 0으로 이어져 표시됨
    Q_lo = min(Q_pos_cell_lo, Q_neg_cell_lo)
    Q_hi = max(Q_pos_cell_hi, Q_neg_cell_hi)
    if Q_hi <= Q_lo:
        return None

    Q_grid = np.linspace(Q_lo, Q_hi, n_grid)

    # ── 각 전극의 (1/m)·dV/dq 를 Q_grid 위에서 계산 ──────────
    dvdq_pos_scaled = _dvdq_on_Q_grid(
        Q_pos, V_pos, pos_mass, pos_slip, Q_grid,
        interp_n=interp_n, smooth_w=smooth_w, dvdq_clip=dvdq_clip)

    dvdq_neg_scaled = _dvdq_on_Q_grid(
        Q_neg, V_neg, neg_mass, neg_slip, Q_grid,
        interp_n=interp_n, smooth_w=smooth_w, dvdq_clip=dvdq_clip)

    # 음극 dV/dQ에 절댓값 적용 (방전 곡선은 음수 기울기 → 양수로 변환)
    dvdq_neg_scaled = np.abs(dvdq_neg_scaled)

    # ── Estimation: dV/dQ = pos항 + |neg항| ──────────────────
    dvdq_est = dvdq_pos_scaled + dvdq_neg_scaled

    # 추가 스무딩 제거: 각 전극에서 이미 스무딩이 적용되었으므로 합산 후 재스무딩 불필요
    dvdq_est = np.clip(dvdq_est, -dvdq_clip, dvdq_clip)

    # ── V-Q 표시용: 각 전극 V를 전지 Q 축에 보간 ──────────────
    # Dahn 정의: Q_cell = q / mass + slip
    Q_pos_cell = Q_pos / pos_mass + pos_slip
    Q_neg_cell = Q_neg / neg_mass + neg_slip

    f_vp = interp1d(Q_pos_cell, V_pos, kind='linear',
                    bounds_error=False, fill_value=np.nan)
    f_vn = interp1d(Q_neg_cell, V_neg, kind='linear',
                    bounds_error=False, fill_value=np.nan)

    V_pos_on_grid = f_vp(Q_grid)
    V_neg_on_grid = f_vn(Q_grid)

    # Estimated Full Cell V-Q:  V_full = V_pos(Q) - V_neg(Q)
    V_full_est = V_pos_on_grid - V_neg_on_grid

    return dict(
        Q_grid             = Q_grid,
        dvdq_pos_scaled    = dvdq_pos_scaled,
        dvdq_neg_scaled    = dvdq_neg_scaled,
        dvdq_estimated     = dvdq_est,
        Q_pos_cell         = (Q_pos_cell_lo, Q_pos_cell_hi),
        Q_neg_cell         = (Q_neg_cell_lo, Q_neg_cell_hi),
        V_pos_on_grid      = V_pos_on_grid,
        V_neg_on_grid      = V_neg_on_grid,
        V_full_est         = V_full_est,
    )


# ══════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════
#  DVA 캔버스 (4 패널)
# ══════════════════════════════════════════════════════════════
class DVACanvas(FigureCanvas):
    """
    2x2 레이아웃:
      [0,0] V-Q Overlay  [0,1] dV/dQ Overlay
      [1,0] Pos scaled   [1,1] Neg scaled

    인터랙션:
      - 마우스 휠: 해당 서브플롯 확대/축소
      - 좌클릭 드래그: 패닝
      - 더블클릭: 해당 축 자동범위 복구
    """
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(14 * UI_SCALE, 9 * UI_SCALE), facecolor=BG_DARK)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._setup_axes()

        # 드래그 상태
        self._drag_ax    = None
        self._drag_start = None   # (x_data, y_data)
        self._drag_xlim  = None
        self._drag_ylim  = None

        # 이벤트 연결
        self.mpl_connect('scroll_event',          self._on_scroll)
        self.mpl_connect('button_press_event',    self._on_press)
        self.mpl_connect('motion_notify_event',   self._on_motion)
        self.mpl_connect('button_release_event',  self._on_release)

        # 자동범위 저장 (render 후 채워짐)
        self._auto_limits = {}   # ax → (xlim, ylim)
        self._param_ax2   = None  # 파라미터 그래프 우측 y축 (twinx)

    def _setup_axes(self):
        # 상단: dV/dQ 전체 너비, 하단: V-Q(좌) + Ref overlay(우)
        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               wspace=0.35, hspace=0.44,
                               left=0.06, right=0.93,
                               top=0.93, bottom=0.08,
                               height_ratios=[1.1, 0.9])
        self.ax_dvdq    = self.fig.add_subplot(gs[0, :])   # 상단 전체 폭
        self.ax_vq      = self.fig.add_subplot(gs[1, 0])   # 하단 좌: V-Q
        self.ax_ref     = self.fig.add_subplot(gs[1, 1])   # 하단 우: Ref overlay
        # 하위 호환을 위해 pos_ref / neg_ref 도 같은 축으로 지정
        self.ax_pos_ref = self.ax_ref
        self.ax_neg_ref = self.ax_ref
        self._axes = [self.ax_dvdq, self.ax_vq, self.ax_ref]
        self._style_all()

    def _style_all(self):
        specs = [
            (self.ax_dvdq,
             "dV/dQ  Overlay  (Measured vs Estimated)",
             "Capacity (mAh)", "dV/dQ  (V/mAh)"),
            (self.ax_vq,
             "V\u2013Q  Overlay  (Full + Pos + Neg)",
             "Capacity (mAh)", "Voltage (V)"),
            (self.ax_ref,
             "Fitting Parameters  vs  Full Cell Cycle",
             "Full Cell Cycle", "Mass  (mp / mn)"),
        ]
        for ax, title, xl, yl in specs:
            self._style_single_ax(ax, title, xl, yl)

    def _legend(self, ax):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=7.5, loc="best",
                      facecolor="#1C1C2E", edgecolor="#3A3A5A",
                      labelcolor="white", framealpha=0.9,
                      borderpad=0.5, handlelength=1.6)

    # ─── 자동 zoom: 양끝 튀는 값 제거 ──────────────────────────
    @staticmethod
    def _trim_limits(data_list, q_lo=3, q_hi=97, margin=0.05):
        """여러 배열에서 하위/상위 q%를 잘라낸 범위 반환.
        0 값은 axis 범위 계산에서 제외 (보간 경계 아티팩트 방지)."""
        arrays = [d.ravel() for d in data_list if d is not None and len(d) > 0]
        if not arrays:
            return None, None
        all_vals = np.concatenate(arrays)
        # 0 제외 시도, 없으면 원본 그대로 사용
        nonzero = all_vals[np.isfinite(all_vals) & (all_vals != 0)]
        all_vals = nonzero if len(nonzero) > 0 else all_vals[np.isfinite(all_vals)]
        if len(all_vals) == 0:
            return None, None
        lo = np.percentile(all_vals, q_lo)
        hi = np.percentile(all_vals, q_hi)
        span = hi - lo if hi > lo else 1.0
        return lo - margin * span, hi + margin * span

    def _apply_auto_zoom(self, ax, xdata_list, ydata_list,
                         x_q=(2, 98), y_q=(1, 99)):
        """x/y 각각 분위수 기반으로 축 범위 설정."""
        xl = self._trim_limits(xdata_list, x_q[0], x_q[1])
        yl = self._trim_limits(ydata_list, y_q[0], y_q[1])
        if xl[0] is not None: ax.set_xlim(*xl)
        if yl[0] is not None: ax.set_ylim(*yl)

    # ─── 마우스 이벤트 ──────────────────────────────────────────
    def _ax_at(self, event):
        """이벤트 좌표에 해당하는 서브플롯 반환."""
        for ax in self._axes:
            if ax.get_visible() and ax.contains_point((event.x, event.y)):
                return ax
        return None

    def _on_scroll(self, event):
        ax = self._ax_at(event)
        if ax is None or event.xdata is None:
            return
        factor = 0.85 if event.button == "up" else 1.0 / 0.85
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        cx, cy = event.xdata, event.ydata
        ax.set_xlim(cx + (xlo - cx) * factor, cx + (xhi - cx) * factor)
        ax.set_ylim(cy + (ylo - cy) * factor, cy + (yhi - cy) * factor)
        self.draw_idle()

    def _on_press(self, event):
        if event.button != 1 or event.xdata is None:
            return
        ax = self._ax_at(event)
        if ax is None:
            return
        # 더블클릭: 자동범위 복구
        if event.dblclick:
            if ax in self._auto_limits:
                xl, yl = self._auto_limits[ax]
                ax.set_xlim(*xl); ax.set_ylim(*yl)
                self.draw_idle()
            return
        # 싱글클릭: 드래그 시작
        self._drag_ax    = ax
        self._drag_start = (event.xdata, event.ydata)
        self._drag_xlim  = ax.get_xlim()
        self._drag_ylim  = ax.get_ylim()

    def _on_motion(self, event):
        if self._drag_ax is None or event.xdata is None:
            return
        dx = event.xdata - self._drag_start[0]
        dy = event.ydata - self._drag_start[1]
        ax = self._drag_ax
        xl = self._drag_xlim
        yl = self._drag_ylim
        ax.set_xlim(xl[0] - dx, xl[1] - dx)
        ax.set_ylim(yl[0] - dy, yl[1] - dy)
        self.draw_idle()

    def _on_release(self, event):
        self._drag_ax = None

    # ─── 렌더 ───────────────────────────────────────────────────
    def render(self, full_r, est, pos_seg, neg_seg,
               full_label="Full Cell",
               pos_label="Positive Ref.",
               neg_label="Negative Ref.",
               fit_range=None,
               pos_mass=1.0, pos_slip=0.0,
               neg_mass=1.0, neg_slip=0.0,
               records=None,
               show_full_dvdq=False):
        """
        fit_range: (q_lo, q_hi) 이면 dV/dQ Overlay에 fitting 범위 음영 표시
        pos_mass, pos_slip, neg_mass, neg_slip: V-Q 패널에서 반쪽전지 Q 축 변환에 사용.
        records: 저장된 피팅 레코드 dict → 우측 하단 파라미터 이력 그래프에 사용.
        show_full_dvdq: True면 fit range 밖에도 Estimated 라인 전체 표시.
        파라미터 변경 시마다 axis를 항상 재계산 (고정 방지).
        """
        for ax in self._axes:
            ax.cla()
        self._style_all()
        self._auto_limits.clear()   # 매 render마다 초기화 → 항상 auto zoom 적용

        COL_FULL = ACCENT_BLUE
        COL_EST  = "#EF233C"
        COL_POS  = "#F4A261"
        COL_NEG  = "#06D6A0"

        # ── 패널 1: V-Q Overlay ──────────────────────────────
        # Full Cell: 변환 없이 원본 그대로
        self.ax_vq.plot(full_r["Q_raw"], full_r["V_raw"],
                        color=COL_FULL, lw=2.0, label=full_label, alpha=0.95)

        vq_x, vq_y = [full_r["Q_raw"]], [full_r["V_raw"]]

        # 반쪽전지: Q_cell = (Q_half - Q_min) / mass + slip 변환 적용 (Dahn 정의)
        def _ref_vq_transformed(seg, mass, slip, color, label):
            if seg is None or "capacity" not in seg.columns:
                return
            Q_h = seg["capacity"].values.astype(float)
            V   = seg["voltage"].values.astype(float)
            ok  = ~(np.isnan(Q_h) | np.isnan(V))
            Q_h, V = Q_h[ok], V[ok]
            if len(Q_h) < 2:
                return
            Q_cell = (Q_h - Q_h.min()) / mass + slip
            idx = np.argsort(Q_cell)
            self.ax_vq.plot(Q_cell[idx], V[idx],
                            color=color, lw=1.4, ls="--",
                            label=label, alpha=0.75)
            vq_x.append(Q_cell[idx]); vq_y.append(V[idx])

        _ref_vq_transformed(pos_seg, pos_mass, pos_slip, COL_POS, "Est. Pos.")
        _ref_vq_transformed(neg_seg, neg_mass, neg_slip, COL_NEG, "Est. Neg.")

        # ── Estimated Full Cell V-Q:  V_pos(Q) − V_neg(Q) ──────
        if est is not None:
            _ok_vf = np.isfinite(est["V_full_est"])
            if _ok_vf.any():
                self.ax_vq.plot(
                    est["Q_grid"][_ok_vf], est["V_full_est"][_ok_vf],
                    color=COL_EST, lw=2.0, ls="--",
                    label="Est. Full Cell V-Q", alpha=0.88)
                vq_x.append(est["Q_grid"][_ok_vf])
                vq_y.append(est["V_full_est"][_ok_vf])

        self._legend(self.ax_vq)
        self._apply_auto_zoom(self.ax_vq, vq_x, vq_y,
                              x_q=(1, 99), y_q=(0.5, 99.5))
        self._auto_limits[self.ax_vq] = (self.ax_vq.get_xlim(),
                                         self.ax_vq.get_ylim())

        # ── 패널 2: dV/dQ Overlay ────────────────────────────
        # 0인 구간 제외
        _qi_full = full_r["Q_i"]
        _dv_full = full_r["dvdq_s"]
        _nonzero_full = _dv_full != 0
        dv_x = [_qi_full[_nonzero_full]]
        dv_y = [_dv_full[_nonzero_full]]

        self.ax_dvdq.plot(_qi_full[_nonzero_full], _dv_full[_nonzero_full],
                          color=COL_FULL, lw=2.2,
                          label="Measured", alpha=0.95)

        if est is not None:
            Q_g   = est["Q_grid"]
            dv_e  = est["dvdq_estimated"]
            dv_p  = est["dvdq_pos_scaled"]
            dv_n  = est["dvdq_neg_scaled"]

            # Estimated 라인: show_full_dvdq=True면 전체, False면 fit_range 내부만
            if show_full_dvdq or fit_range is None:
                _in_range = np.ones(len(Q_g), dtype=bool)
            else:
                q_lo, q_hi = fit_range
                _in_range = (Q_g >= q_lo) & (Q_g <= q_hi)

            _nz_est = (dv_e != 0) & _in_range
            _nz_pos = (dv_p != 0) & _in_range
            _nz_neg = (dv_n != 0) & _in_range

            if _nz_est.any():
                self.ax_dvdq.plot(Q_g[_nz_est], dv_e[_nz_est],
                                  color=COL_EST, lw=2.0, ls="--",
                                  label="Estimated", alpha=0.88)
            if _nz_pos.any():
                self.ax_dvdq.plot(Q_g[_nz_pos], dv_p[_nz_pos],
                                  color=COL_POS, lw=1.2, ls=":",
                                  label="(1/mp)dVp", alpha=0.65)
            if _nz_neg.any():
                self.ax_dvdq.plot(Q_g[_nz_neg], dv_n[_nz_neg],
                                  color=COL_NEG, lw=1.2, ls=":",
                                  label="(1/mn)dVn", alpha=0.65)

            # auto zoom 범위 계산에는 전체 범위 사용
            _nz_all = dv_e != 0
            dv_x += [Q_g[_nz_all], Q_g[dv_p != 0], Q_g[dv_n != 0]]
            dv_y += [dv_e[_nz_all], dv_p[dv_p != 0], dv_n[dv_n != 0]]

        self.ax_dvdq.axhline(0, color="#FFFFFF35", lw=0.8, ls="--", zorder=0)

        # ── Fitting 범위 표시: axvspan만, axvline 없음 ────────
        if fit_range is not None:
            q_lo, q_hi = fit_range
            self.ax_dvdq.axvspan(q_lo, q_hi,
                                 alpha=0.12, color="#EF233C", zorder=0,
                                 label=f"Fit range [{q_lo:.2f}–{q_hi:.2f}]")
            # V-Q 패널: axvspan만 (axvline 제거)
            self.ax_vq.axvspan(q_lo, q_hi,
                               alpha=0.08, color="#EF233C", zorder=0)

        self._legend(self.ax_dvdq)
        self._apply_auto_zoom(self.ax_dvdq, dv_x, dv_y,
                              x_q=(2, 98), y_q=(2, 98))
        self._auto_limits[self.ax_dvdq] = (self.ax_dvdq.get_xlim(),
                                           self.ax_dvdq.get_ylim())

        # ── 패널 3: 파라미터 이력 그래프 ────────────────────────
        self.render_param_history(records if records is not None else {}, _draw=False)

        self.draw()

    def _style_single_ax(self, ax, title, xlabel, ylabel):
        """단일 축에 공통 다크 스타일 적용."""
        ax.set_facecolor("#1A1A2E")
        ax.tick_params(colors="#AAAACC", labelsize=8, length=3)
        ax.set_xlabel(xlabel, color="#AAAACC", fontsize=9, labelpad=4)
        ax.set_ylabel(ylabel, color="#AAAACC", fontsize=9, labelpad=4)
        ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2A2A4A"); sp.set_linewidth(0.8)
        ax.grid(True, color="#FFFFFF10", linewidth=0.5, linestyle="--")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="minor", colors="#FFFFFF20", length=2)

    def _style_param_ax(self):
        """파라미터 이력 축 기본 스타일 적용 (데이터 없을 때도 동일하게)."""
        ax = self.ax_ref
        self._style_single_ax(ax,
            "Fitting Parameters  vs  Full Cell Cycle",
            "Full Cell Cycle", "Mass  (mp / mn)")
        if not ax.get_lines() and not ax.collections:
            ax.text(0.5, 0.5, "Save fitting records to display",
                    transform=ax.transAxes,
                    color="#444466", fontsize=9, ha='center', va='center',
                    style='italic')

    def render_param_history(self, records: dict, _draw=True):
        """
        저장된 레코드 딕셔너리로 파라미터 이력 그래프를 그린다.

        x축: Full Cell 사이클 번호
        좌측 y축: mp (양극 mass), mn (음극 mass)
        우측 y축: δp (양극 slippage), δn (음극 slippage)

        동일 사이클에 복수 direction이 있으면 평균을 취한다.
        """
        # ── 기존 twinx 축 완전 제거 후 ax_ref 초기화 ────────────
        if self._param_ax2 is not None:
            try:
                self._param_ax2.remove()
            except Exception:
                pass
            self._param_ax2 = None

        ax = self.ax_ref
        ax.cla()
        self._style_param_ax()

        if not records:
            if _draw: self.draw_idle()
            return

        # ── 레코드 파싱: Full Cell 사이클별로 집계 ──────────────
        bucket = defaultdict(list)   # full_cyc → [record, ...]
        for rec in records.values():
            fc = rec.get('full_cyc', None)
            if fc is not None:
                bucket[int(fc)].append(rec)

        cycles = sorted(bucket.keys())
        if not cycles:
            if _draw: self.draw_idle()
            return

        # 같은 사이클 내 복수 레코드는 평균
        def _mean(recs, key):
            vals = [r[key] for r in recs if key in r and np.isfinite(r[key])]
            return float(np.mean(vals)) if vals else np.nan

        cyc_arr = np.array(cycles)
        mp_arr  = np.array([_mean(bucket[c], 'mp') for c in cycles])
        mn_arr  = np.array([_mean(bucket[c], 'mn') for c in cycles])
        dp_arr  = np.array([_mean(bucket[c], 'dp') for c in cycles])
        dn_arr  = np.array([_mean(bucket[c], 'dn') for c in cycles])

        COL_MP  = "#F4A261"   # 양극 mass — 오렌지
        COL_MN  = "#4CC9F0"   # 음극 mass — 블루
        COL_DP  = "#F4A26188" # 양극 slip — 오렌지 반투명
        COL_DN  = "#4CC9F088" # 음극 slip — 블루 반투명
        MS      = 6           # marker size

        # ── 좌측 y축: mass (mp, mn) ──────────────────────────────
        ax.set_ylabel("Mass  (mp / mn)", color="#AAAACC", fontsize=9, labelpad=4)

        valid_mp = np.isfinite(mp_arr)
        valid_mn = np.isfinite(mn_arr)
        if valid_mp.any():
            ax.plot(cyc_arr[valid_mp], mp_arr[valid_mp],
                    color=COL_MP, lw=2.0, marker='o', ms=MS,
                    label="mp (pos mass)", zorder=3)
        if valid_mn.any():
            ax.plot(cyc_arr[valid_mn], mn_arr[valid_mn],
                    color=COL_MN, lw=2.0, marker='s', ms=MS,
                    label="mn (neg mass)", zorder=3)

        # ── 우측 y축: slippage (dp, dn) ──────────────────────────
        ax2 = ax.twinx()
        ax2.set_facecolor("none")
        ax2.tick_params(colors="#AAAACC", labelsize=8, length=3)
        ax2.set_ylabel("Slippage  δ  (mAh)", color="#AAAACC",
                       fontsize=9, labelpad=4)
        for sp in ax2.spines.values():
            sp.set_edgecolor("#2A2A4A"); sp.set_linewidth(0.8)

        valid_dp = np.isfinite(dp_arr)
        valid_dn = np.isfinite(dn_arr)
        if valid_dp.any():
            ax2.plot(cyc_arr[valid_dp], dp_arr[valid_dp],
                     color=COL_MP, lw=1.4, ls="--", marker='^', ms=MS - 1,
                     alpha=0.75, label="δp (pos slip)", zorder=2)
        if valid_dn.any():
            ax2.plot(cyc_arr[valid_dn], dn_arr[valid_dn],
                     color=COL_MN, lw=1.4, ls="--", marker='v', ms=MS - 1,
                     alpha=0.75, label="δn (neg slip)", zorder=2)

        # x축 정수 눈금
        if len(cycles) <= 20:
            ax.set_xticks(cyc_arr)
        ax.set_xlim(cyc_arr[0] - 0.5, cyc_arr[-1] + 0.5)

        # ── 통합 범례 ─────────────────────────────────────────────
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h1 or h2:
            ax.legend(h1 + h2, l1 + l2,
                      fontsize=7.5, loc="best",
                      facecolor="#1C1C2E", edgecolor="#3A3A5A",
                      labelcolor="white", framealpha=0.9,
                      borderpad=0.5, handlelength=1.6)

        # ax2를 _axes 목록에서 제외 (드래그/줌 이벤트 오동작 방지)
        self._param_ax2 = ax2   # 참조 보관 (cla 시 정리용)

        self._auto_limits[ax] = (ax.get_xlim(), ax.get_ylim())
        if _draw:
            self.draw_idle()



# ══════════════════════════════════════════════════════════════
#  파일 로드 + 사이클/방향 선택 카드
# ══════════════════════════════════════════════════════════════
class FileCard(QGroupBox):
    sig_changed = pyqtSignal()

    def __init__(self, key, label, color, parent=None):
        super().__init__(label, parent)
        self.key   = key
        self.color = color
        self.df    = None
        self.segs  = {}
        self.setStyleSheet(
            f"QGroupBox {{"
            f"  color:{color};"
            f"  border:2px solid {color};"
            f"  border-radius:{px(8)}px; margin-top:{px(12)}px; padding:{px(4)}px;"
            f"  font-weight:700;"
            f"  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"    stop:0 {color}18, stop:1 {BG_PANEL});"
            f"}}"
            f"QGroupBox::title {{"
            f"  color:{color}; subcontrol-origin:margin;"
            f"  left:{px(10)}px; top:-2px; padding:0 {px(6)}px;"
            f"  font-size:{px(11)}px; font-weight:800;"
            f"  background:{BG_PANEL};"
            f"}}")
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(4)
        lay.setContentsMargins(6, 6, 6, 6)

        # ── 행1: 파일명 레이블(좌) + 정사각형 Open 버튼(우) ──
        top_row = QHBoxLayout()
        top_row.setSpacing(5)
        top_row.setContentsMargins(0, 0, 0, 0)

        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setWordWrap(False)
        self.lbl_file.setMinimumWidth(0)
        self.lbl_file.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_file.setFixedHeight(px(34))
        self.lbl_file.setStyleSheet(
            f"color:{TEXT_SECONDARY}; font-size:{px(9)}px; padding:2px {px(6)}px;"
            f" background:{BG_CARD}; border-radius:{px(4)}px; font-weight:400;")
        top_row.addWidget(self.lbl_file, 1)

        # 정사각형 Open 버튼 (폴더 아이콘)
        btn = QPushButton("📂")
        btn.setFixedSize(px(34), px(34))
        btn.setToolTip("Open File  (MPT / CSV / TXT / XLSX)")
        btn.setStyleSheet(
            f"QPushButton {{"
            f"  background:{self.color}33; border:2px solid {self.color}88;"
            f"  color:{self.color}; border-radius:{px(6)}px;"
            f"  font-size:{px(16)}px; padding:0;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background:{self.color}55; border-color:{self.color};"
            f"}}"
            f"QPushButton:pressed {{"
            f"  background:{self.color}77;"
            f"}}")
        btn.clicked.connect(self._open_file)
        top_row.addWidget(btn)
        lay.addLayout(top_row)

        # ── 행2: Cycle + Dir ──────────────────────────────────
        row = QHBoxLayout()
        row.setSpacing(5)
        row.setContentsMargins(0, 0, 0, 0)

        lbl_c = QLabel("Cycle:")
        lbl_c.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:{px(10)}px; font-weight:400;")
        lbl_c.setFixedWidth(px(36))
        row.addWidget(lbl_c)

        self.spin_cyc = QSpinBox()
        self.spin_cyc.setRange(1, 9999)
        self.spin_cyc.setValue(1)
        self.spin_cyc.setFixedWidth(px(56))
        self.spin_cyc.setFixedHeight(px(24))
        self.spin_cyc.valueChanged.connect(self.sig_changed)
        row.addWidget(self.spin_cyc)

        row.addSpacing(4)
        lbl_d = QLabel("Dir:")
        lbl_d.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:{px(10)}px; font-weight:400;")
        lbl_d.setFixedWidth(px(26))
        row.addWidget(lbl_d)

        self.cmb_dir = QComboBox()
        self.cmb_dir.addItems(["Charge ↑", "Discharge ↓"])
        self.cmb_dir.setFixedHeight(px(24))
        self.cmb_dir.currentIndexChanged.connect(self.sig_changed)
        row.addWidget(self.cmb_dir, 1)
        lay.addLayout(row)

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "",
            "EC-Lab MPT (*.mpt);;Excel (*.xlsx *.xls);;CSV / TSV / TXT (*.csv *.txt *.tsv);;All Files (*)")
        if not path:
            return
        try:
            df = load_data(path)
            df._filename = path
            self.df   = df
            self.segs = split_cycles(df)
            cycles    = sorted(df['cycle'].unique())
            self.spin_cyc.blockSignals(True)
            self.spin_cyc.setRange(cycles[0], cycles[-1])
            self.spin_cyc.setValue(cycles[0])
            self.spin_cyc.blockSignals(False)
            ftype = getattr(df, '_file_type', 'csv').upper()
            fname = os.path.basename(path)
            if len(fname) > 28:
                fname = '…' + fname[-26:]
            cyc_info = f"Cyc {cycles[0]}–{cycles[-1]}"
            self.lbl_file.setText(
                f'[{ftype}] {fname}  |  {cyc_info}')
            self.lbl_file.setTextFormat(0)   # PlainText (wordWrap False와 호환)
            self.lbl_file.setStyleSheet(
                f"color:{self.color}; font-size:9px; padding:2px 6px;"
                f" background:{BG_CARD}; border-radius:4px; font-weight:600;")
            self.lbl_file.setToolTip(
                f"{path}\n{len(df):,} rows · {df['cycle'].nunique()} cycles")
            self.sig_changed.emit()
        except Exception as e:
            QMessageBox.critical(self, "File Error", str(e))

    def get_segment(self):
        """(segment_df | None, cycle_int, direction_str) 반환."""
        if self.df is None:
            return None, None, None
        cyc = self.spin_cyc.value()
        direction = 'charge' if self.cmb_dir.currentIndex() == 0 else 'discharge'
        return self.segs.get((cyc, direction)), cyc, direction


# ══════════════════════════════════════════════════════════════
#  DVA 메인 패널 (좌측 컨트롤 + 우측 캔버스)
# ══════════════════════════════════════════════════════════════
class DVAPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._recompute)

        # 슬라이더 전용 타이머 (V-Q만 빠르게 갱신)
        self._vq_timer = QTimer()
        self._vq_timer.setSingleShot(True)
        self._vq_timer.timeout.connect(self._refresh_all)
        self._labels_cache = ("Full Cell", "Positive Ref.", "Negative Ref.")
        self._fit_thread = None   # 진행 중인 피팅 스레드
        self._full_r_cache = None  # 마지막 계산된 full_r
        self._est_cache = None     # 마지막 계산된 estimated 결과 캐시
        self._est_cache_key = None # estimated 캐시 무효화 키
        self._pos_seg_cache = None
        self._neg_seg_cache = None
        # ── 임시 저장소: key = "Full{fc}{fd}_Pos{pc}{pd}_Neg{nc}{nd}", value = record dict
        self._saved_records = {}   # OrderedDict처럼 삽입 순서 유지 (Python 3.7+)
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 좌측 컨트롤 ─────────────────────────────────────
        ctrl_w = QWidget()
        ctrl_lay = QVBoxLayout(ctrl_w)
        ctrl_lay.setContentsMargins(12, 10, 16, 10)
        ctrl_lay.setSpacing(8)

        t = QLabel("DVA Tool_v16")
        t.setStyleSheet(
            f"font-size:{px(15)}px; font-weight:800;"
            f" color:{ACCENT_BLUE}; letter-spacing:2px;")
        s = QLabel("Differential Voltage Analysis")
        s.setStyleSheet(f"font-size:{px(10)}px; color:{TEXT_SECONDARY};")
        ctrl_lay.addWidget(t)
        ctrl_lay.addWidget(s)
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        ctrl_lay.addWidget(sep)

        # ════════════════════════════════════════════════════
        #  공통 유틸  ─ 화살표 버튼 + SpinBox 파라미터 행 생성
        # ════════════════════════════════════════════════════

        def _make_spin_style(color):
            return (
                f"QDoubleSpinBox, QSpinBox {{"
                f"  background:#12122A; color:{color};"
                f"  border:2px solid {color}; border-radius:{px(6)}px;"
                f"  padding:{px(3)}px {px(8)}px; font-weight:800; font-size:{px(13)}px;"
                f"  min-width:{px(90)}px; }}"
                f"QDoubleSpinBox::up-button, QSpinBox::up-button,"
                f"QDoubleSpinBox::down-button, QSpinBox::down-button {{"
                f"  width:0px; border:none; }}"  # 기본 화살표 숨김
            )

        def _arrow_btn_style(color):
            return (
                f"QPushButton {{"
                f"  background:{color}22; border:2px solid {color}66;"
                f"  border-radius:{px(6)}px; color:{color};"
                f"  font-size:{px(15)}px; font-weight:900;"
                f"  min-width:{px(32)}px; max-width:{px(32)}px;"
                f"  min-height:{px(32)}px; max-height:{px(32)}px; padding:0; }}"
                f"QPushButton:hover {{"
                f"  background:{color}55; border-color:{color}; }}"
                f"QPushButton:pressed {{"
                f"  background:{color}88; }}"
            )

        def _param_row(parent_lay, label_txt, lo, hi, val, dec, step,
                       color, is_int=False, vq_only=False, compact=False):
            """
            normal  : [레이블 줄] + [◀◀ ◀ SpinBox ▶ ▶▶ 줄]  (2행, 높이 62)
            compact : [레이블 | ◀◀ ◀ SpinBox ▶ ▶▶]  (1행, 높이 30)
            """
            container = QWidget()
            container.setFixedHeight(px(30) if compact else px(62))

            if compact:
                # 1행 레이아웃: 레이블 + 버튼 + spin + 버튼
                hlay = QHBoxLayout(container)
                hlay.setContentsMargins(2, 0, 2, 0)
                hlay.setSpacing(3)

                lbl = QLabel(label_txt)
                lbl.setStyleSheet(
                    f"color:{color}; font-size:{px(9)}px; font-weight:700;")
                lbl.setFixedWidth(px(52))
                hlay.addWidget(lbl)
            else:
                # 2행 레이아웃
                vlay = QVBoxLayout(container)
                vlay.setContentsMargins(4, 2, 4, 2)
                vlay.setSpacing(3)
                lbl = QLabel(label_txt)
                lbl.setStyleSheet(
                    f"color:{color}; font-size:{px(11)}px; font-weight:700;")
                vlay.addWidget(lbl)
                hlay = QHBoxLayout()
                hlay.setContentsMargins(0, 0, 0, 0)
                hlay.setSpacing(4)
                vlay.addLayout(hlay)

            if is_int:
                spin = QSpinBox()
                spin.setRange(int(lo), int(hi))
                spin.setValue(int(val))
                spin.setSingleStep(int(step))
                big_step = int(step) * 10
            else:
                spin = QDoubleSpinBox()
                spin.setRange(lo, hi)
                spin.setValue(val)
                spin.setDecimals(dec)
                spin.setSingleStep(step)
                big_step = step * 10

            spin.setFixedHeight(px(24) if compact else px(34))
            spin.setMaximumWidth(px(110) if compact else px(130))
            spin.setMinimumWidth(px(70) if compact else px(80))
            spin.setStyleSheet(_make_spin_style(color))

            btn_ll = QPushButton("◀◀")
            btn_l  = QPushButton("◀")
            btn_r  = QPushButton("▶")
            btn_rr = QPushButton("▶▶")
            if compact:
                _btn_size = (
                    f"QPushButton {{"
                    f"  background:{color}22; border:1px solid {color}55;"
                    f"  border-radius:{px(4)}px; color:{color};"
                    f"  font-size:{px(10)}px; font-weight:900;"
                    f"  min-width:{px(22)}px; max-width:{px(22)}px;"
                    f"  min-height:{px(22)}px; max-height:{px(22)}px; padding:0; }}"
                    f"QPushButton:hover {{"
                    f"  background:{color}44; border-color:{color}; }}"
                    f"QPushButton:pressed {{ background:{color}66; }}"
                )
                for b in (btn_ll, btn_l, btn_r, btn_rr):
                    b.setStyleSheet(_btn_size)
            else:
                for b in (btn_ll, btn_l, btn_r, btn_rr):
                    b.setStyleSheet(_arrow_btn_style(color))

            for b in (btn_ll, btn_l, btn_r, btn_rr):
                b.setAutoRepeat(True)
                b.setAutoRepeatDelay(400)
                b.setAutoRepeatInterval(80)

            hlay.addWidget(btn_ll)
            hlay.addWidget(btn_l)
            if not compact:
                hlay.addStretch(1)
            hlay.addWidget(spin)
            if not compact:
                hlay.addStretch(1)
            hlay.addWidget(btn_r)
            hlay.addWidget(btn_rr)
            if compact:
                hlay.addStretch(1)

            parent_lay.addWidget(container)

            # ── 버튼 콜백 ──────────────────────────────────────
            def _change(delta, _sp=spin, _ii=is_int, _vq=vq_only):
                if _ii:
                    new_v = int(np.clip(_sp.value() + delta, lo, hi))
                else:
                    new_v = float(np.clip(_sp.value() + delta, lo, hi))
                _sp.setValue(new_v)

            btn_ll.clicked.connect(lambda: _change(-big_step))
            btn_l .clicked.connect(lambda: _change(-step))
            btn_r .clicked.connect(lambda: _change( step))
            btn_rr.clicked.connect(lambda: _change( big_step))

            # SpinBox 직접 편집 시 재계산
            # vq_only=True (mass/slip): 세그먼트는 그대로, 파라미터만 바뀌므로
            #   캐시 데이터로 전체 3패널을 즉시 재렌더 (full recompute 불필요)
            # vq_only=False (smoothing/interp 등): 전체 재계산 필요
            def _on_spin(v, _vq=vq_only):
                if _vq:
                    self._schedule_vq()
                else:
                    self._schedule()

            spin.valueChanged.connect(_on_spin)
            return spin, None   # sld 자리는 None 반환 (호환 유지)

        def _grp_box(title, color):
            g = QGroupBox(title)
            g.setStyleSheet(
                f"QGroupBox {{ color:{color}; border:2px solid {color};"
                f" border-radius:{px(8)}px; margin-top:{px(14)}px; padding:{px(6)}px {px(4)}px; }}"
                f"QGroupBox::title {{ color:{color}; font-weight:800;"
                f" font-size:{px(11)}px; subcontrol-origin:margin;"
                f" left:{px(10)}px; top:-2px; padding:0 {px(6)}px;"
                f" background:{BG_PANEL}; }}")
            lay = QVBoxLayout(g)
            lay.setSpacing(4)
            lay.setContentsMargins(6, 8, 6, 6)
            return g, lay

        # ════════════════════════════════════════════════════
        #  PROCESSING 그룹  (파일카드 바로 아래)
        # ════════════════════════════════════════════════════
        grp_proc, proc_lay = _grp_box("PROCESSING", ACCENT_PURPLE)
        proc_lay.setSpacing(2)

        lbl_s = QLabel("Smoothing  (Savitzky-Golay)")
        lbl_s.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:{px(9)}px; font-weight:600;")
        proc_lay.addWidget(lbl_s)
        self.sp_smooth, _ = _param_row(
            proc_lay, "Window", 3, 101, 7, 0, 2,
            ACCENT_BLUE, is_int=True, compact=True)

        lbl_i = QLabel("Interpolation points")
        lbl_i.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:{px(9)}px; font-weight:600;")
        proc_lay.addWidget(lbl_i)
        self.sp_interp, _ = _param_row(
            proc_lay, "Points", 200, 5000, 1000, 0, 100,
            ACCENT_PURPLE, is_int=True, compact=True)

        ctrl_lay.addWidget(grp_proc)

        # ════════════════════════════════════════════════════
        #  3개 파일 카드
        # ════════════════════════════════════════════════════
        self.card_full = FileCard('full', 'Full Cell',     ACCENT_BLUE)
        self.card_pos  = FileCard('pos',  'Positive Ref.', '#F4A261')
        self.card_neg  = FileCard('neg',  'Negative Ref.', '#06D6A0')
        for card in (self.card_full, self.card_pos, self.card_neg):
            card.sig_changed.connect(self._schedule)
            ctrl_lay.addWidget(card)

        # ════════════════════════════════════════════════════
        #  DVA PARAMETERS  (전극별 mass / slippage)
        # ════════════════════════════════════════════════════
        grp_dva, dva_lay = _grp_box("DVA PARAMETERS", ACCENT_BLUE)

        # ── Positive
        pos_lbl = QLabel("Positive Electrode")
        pos_lbl.setStyleSheet(
            f"color:#F4A261; font-weight:800; font-size:{px(11)}px;"
            f" border-bottom:1px solid #F4A26155; padding-bottom:2px;")
        dva_lay.addWidget(pos_lbl)
        self.sp_pos_mass, self._sld_pos_mass = _param_row(
            dva_lay, "Mass  mp", 0.01, 5.0, 1.0, 4, 0.01, '#F4A261', vq_only=True)
        self.sp_pos_slip, self._sld_pos_slip = _param_row(
            dva_lay, "Slippage  δp  (mAh)", -50.0, 50.0, 0.0, 3, 0.01, '#F4A261', vq_only=True)

        # ── Negative
        neg_lbl = QLabel("Negative Electrode")
        neg_lbl.setStyleSheet(
            f"color:#06D6A0; font-weight:800; font-size:{px(11)}px;"
            f" border-bottom:1px solid #06D6A055; padding-bottom:2px;")
        dva_lay.addWidget(neg_lbl)
        self.sp_neg_mass, self._sld_neg_mass = _param_row(
            dva_lay, "Mass  mn", 0.01, 5.0, 1.0, 4, 0.01, '#06D6A0', vq_only=True)
        self.sp_neg_slip, self._sld_neg_slip = _param_row(
            dva_lay, "Slippage  δn  (mAh)", -50.0, 50.0, 0.0, 3, 0.01, '#06D6A0', vq_only=True)

        ctrl_lay.addWidget(grp_dva)

        # ════════════════════════════════════════════════════
        #  FITTING 그룹
        # ════════════════════════════════════════════════════
        grp_fit, fit_lay = _grp_box("FITTING", '#EF233C')

        # Fitting 용량 범위
        rng_lbl = QLabel("Fitting Q range (mAh)")
        rng_lbl.setStyleSheet(
            f"color:{TEXT_SECONDARY}; font-size:{px(10)}px; font-weight:600;")
        fit_lay.addWidget(rng_lbl)

        rng_row = QHBoxLayout()
        rng_row.setSpacing(6)
        lbl_from = QLabel("From")
        lbl_from.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:{px(10)}px;")
        rng_row.addWidget(lbl_from)

        self.sp_fit_qlo = QDoubleSpinBox()
        self.sp_fit_qlo.setRange(-100, 9999); self.sp_fit_qlo.setValue(0.0)
        self.sp_fit_qlo.setDecimals(2); self.sp_fit_qlo.setSingleStep(0.05)
        self.sp_fit_qlo.setFixedHeight(px(28)); self.sp_fit_qlo.setMinimumWidth(px(75))
        self.sp_fit_qlo.setStyleSheet(
            f"QDoubleSpinBox {{ background:#12122A; color:#EF233C;"
            f" border:2px solid #EF233C; border-radius:{px(5)}px;"
            f" padding:2px {px(6)}px; font-weight:800; font-size:{px(12)}px; }}")
        self.sp_fit_qlo.valueChanged.connect(self._schedule)
        rng_row.addWidget(self.sp_fit_qlo)

        lbl_to = QLabel("To")
        lbl_to.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:{px(10)}px;")
        rng_row.addWidget(lbl_to)

        self.sp_fit_qhi = QDoubleSpinBox()
        self.sp_fit_qhi.setRange(-100, 9999); self.sp_fit_qhi.setValue(9999.0)
        self.sp_fit_qhi.setDecimals(2); self.sp_fit_qhi.setSingleStep(0.05)
        self.sp_fit_qhi.setFixedHeight(px(28)); self.sp_fit_qhi.setMinimumWidth(px(75))
        self.sp_fit_qhi.setStyleSheet(
            f"QDoubleSpinBox {{ background:#12122A; color:#EF233C;"
            f" border:2px solid #EF233C; border-radius:{px(5)}px;"
            f" padding:2px {px(6)}px; font-weight:800; font-size:{px(12)}px; }}")
        self.sp_fit_qhi.valueChanged.connect(self._schedule)
        rng_row.addWidget(self.sp_fit_qhi)
        fit_lay.addLayout(rng_row)

        # Fitting 파라미터 활성화 체크박스
        chk_lbl = QLabel("Fit parameters:")
        chk_lbl.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:{px(10)}px; font-weight:600;")
        fit_lay.addWidget(chk_lbl)

        chk_row = QHBoxLayout()
        chk_row.setSpacing(8)
        self.chk_fit_mp = QCheckBox("mp"); self.chk_fit_mp.setChecked(True)
        self.chk_fit_mn = QCheckBox("mn"); self.chk_fit_mn.setChecked(True)
        self.chk_fit_dp = QCheckBox("δp"); self.chk_fit_dp.setChecked(True)
        self.chk_fit_dn = QCheckBox("δn"); self.chk_fit_dn.setChecked(True)
        for chk in (self.chk_fit_mp, self.chk_fit_mn,
                    self.chk_fit_dp, self.chk_fit_dn):
            chk.setStyleSheet(f"color:{TEXT_PRIMARY}; font-size:{px(11)}px;")
            chk_row.addWidget(chk)
        chk_row.addStretch()
        fit_lay.addLayout(chk_row)

        # ── Fitting 전략 ──────────────────────────────────────
        strat_lbl = QLabel("Optimization strategy")
        strat_lbl.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:{px(10)}px; font-weight:600;")
        fit_lay.addWidget(strat_lbl)

        self.cmb_strategy = QComboBox()
        self.cmb_strategy.addItems([
            "Multi-start + TRF  (Grid scan)  ★ Best",
            "TRF only  (Fast, single start)",
        ])
        self.cmb_strategy.setCurrentIndex(0)
        self.cmb_strategy.setFixedHeight(px(28))
        self.cmb_strategy.setStyleSheet(
            f"QComboBox {{ background:#12122A; color:#EF233C;"
            f" border:2px solid #EF233C66; border-radius:{px(5)}px;"
            f" padding:2px {px(8)}px; font-weight:700; font-size:{px(11)}px; }}"
            f"QComboBox::drop-down {{ border:none; width:{px(16)}px; }}"
            f"QComboBox QAbstractItemView {{ background:{BG_CARD};"
            f" color:{TEXT_PRIMARY}; selection-background-color:{ACCENT_PURPLE}; }}")
        fit_lay.addWidget(self.cmb_strategy)

        # ── 피크 가중치 + 전체 dV/dQ 표시 옵션 ─────────────────
        opt_row = QHBoxLayout(); opt_row.setSpacing(10)
        self.chk_peak_weight = QCheckBox("Peak weighting")
        self.chk_peak_weight.setChecked(True)
        self.chk_peak_weight.setToolTip(
            "dV/dQ 피크 영역에 높은 가중치 부여 → 피크 매칭 품질 향상")
        self.chk_peak_weight.setStyleSheet(f"color:{TEXT_PRIMARY}; font-size:{px(11)}px;")
        opt_row.addWidget(self.chk_peak_weight)

        self.chk_show_full_dvdq = QCheckBox("Show full dV/dQ")
        self.chk_show_full_dvdq.setChecked(False)   # 기본값: fit range만 표시
        self.chk_show_full_dvdq.setToolTip(
            "체크: Estimated dV/dQ를 fitting 범위 밖에도 표시\n"
            "미체크(기본): fitting 범위 내부만 표시")
        self.chk_show_full_dvdq.setStyleSheet(f"color:{TEXT_PRIMARY}; font-size:{px(11)}px;")
        self.chk_show_full_dvdq.stateChanged.connect(self._refresh_all)
        opt_row.addWidget(self.chk_show_full_dvdq)
        opt_row.addStretch()
        fit_lay.addLayout(opt_row)

        # ── Fitting 정밀도 ────────────────────────────────────
        prec_lbl = QLabel("Fitting precision")
        prec_lbl.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:{px(10)}px; font-weight:600;")
        fit_lay.addWidget(prec_lbl)

        prec_row = QHBoxLayout()
        prec_row.setSpacing(6)

        tol_lbl = QLabel("Tolerance")
        tol_lbl.setStyleSheet(f"color:{TEXT_SECONDARY}; font-size:{px(10)}px;")
        prec_row.addWidget(tol_lbl)

        self.cmb_tol = QComboBox()
        self.cmb_tol.addItems(["1e-4", "1e-5", "1e-6", "1e-7", "1e-8", "1e-9", "1e-10",
                                "1e-11", "1e-12", "1e-13", "1e-14", "1e-15"])
        self.cmb_tol.setCurrentIndex(4)   # 기본값 1e-8
        self.cmb_tol.setFixedHeight(px(26))
        self.cmb_tol.setStyleSheet(
            f"QComboBox {{ background:#12122A; color:#EF233C;"
            f" border:2px solid #EF233C66; border-radius:{px(5)}px;"
            f" padding:2px {px(6)}px; font-weight:700; font-size:{px(11)}px; }}"
            f"QComboBox::drop-down {{ border:none; width:{px(16)}px; }}"
            f"QComboBox QAbstractItemView {{ background:{BG_CARD};"
            f" color:{TEXT_PRIMARY}; selection-background-color:{ACCENT_PURPLE}; }}")
        prec_row.addWidget(self.cmb_tol)
        prec_row.addStretch()
        fit_lay.addLayout(prec_row)

        # Fitting 버튼
        self.btn_fit = QPushButton("▶  AUTO FIT")
        self.btn_fit.setMinimumHeight(px(38))
        self.btn_fit.setStyleSheet(
            f"QPushButton {{"
            f"  background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"    stop:0 #C0392B, stop:1 #EF233C);"
            f"  color:white; border:none; border-radius:{px(8)}px;"
            f"  font-size:{px(13)}px; font-weight:900; letter-spacing:1px; }}"
            f"QPushButton:hover {{"
            f"  background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"    stop:0 #E74C3C, stop:1 #FF4D5E); }}"
            f"QPushButton:disabled {{"
            f"  background:#444; color:#888; }}")
        self.btn_fit.clicked.connect(self._run_fit)
        fit_lay.addWidget(self.btn_fit)

        # Fitting 결과 레이블
        self.lbl_fit_result = QLabel("Fitting results will appear here.")
        self.lbl_fit_result.setWordWrap(True)
        self.lbl_fit_result.setStyleSheet(
            f"background:#1A0A0A; border:1px solid #EF233C55;"
            f" border-radius:{px(6)}px; padding:{px(7)}px;"
            f" color:{TEXT_PRIMARY}; font-size:{px(11)}px;")
        fit_lay.addWidget(self.lbl_fit_result)

        ctrl_lay.addWidget(grp_fit)

        # 결과 요약 레이블
        self.lbl_status = QLabel(
            "Load all 3 files to start automatic analysis.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(
            f"background:{BG_CARD}; border:1px solid {BORDER_COLOR};"
            f" border-radius:{px(6)}px; padding:{px(8)}px;"
            f" color:{TEXT_PRIMARY}; font-size:{px(11)}px;")
        ctrl_lay.addWidget(self.lbl_status)

        # ═══════════════════════════════════════════════════════
        #  💾 Save Fit 버튼  —  현재 fitting 데이터를 메모리에 임시 저장
        # ═══════════════════════════════════════════════════════
        self.btn_save_fit = QPushButton("💾  Save Current Fit")
        self.btn_save_fit.setMinimumHeight(px(38))
        self.btn_save_fit.setStyleSheet(
            f"QPushButton {{"
            f"  background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"    stop:0 {ACCENT_PURPLE}, stop:1 #4361EE);"
            f"  color:white; border:none; border-radius:{px(8)}px;"
            f"  font-size:{px(12)}px; font-weight:800; letter-spacing:1px; }}"
            f"QPushButton:hover {{"
            f"  background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"    stop:0 #9B3FDE, stop:1 {ACCENT_BLUE}); }}"
            f"QPushButton:disabled {{ background:#2A2A3A; color:#555566; }}")
        self.btn_save_fit.clicked.connect(self._save_current_fit)
        ctrl_lay.addWidget(self.btn_save_fit)

        # ═══════════════════════════════════════════════════════
        #  저장된 레코드 목록 그룹
        # ═══════════════════════════════════════════════════════
        grp_saved, saved_lay = _grp_box("SAVED RECORDS", '#F4A261')

        self.lbl_saved_count = QLabel("No saved records.")
        self.lbl_saved_count.setStyleSheet(
            f"color:{TEXT_SECONDARY}; font-size:{px(10)}px; font-style:italic;")
        saved_lay.addWidget(self.lbl_saved_count)

        self.list_saved = QListWidget()
        self.list_saved.setMinimumHeight(px(80))
        self.list_saved.setMaximumHeight(px(150))
        self.list_saved.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_saved.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_saved.customContextMenuRequested.connect(self._saved_context_menu)
        self.list_saved.setStyleSheet(
            f"QListWidget {{ background:{BG_CARD}; border:1px solid {BORDER_COLOR};"
            f"  border-radius:{px(5)}px; color:{TEXT_PRIMARY}; font-size:{px(10)}px; }}"
            f"QListWidget::item {{ padding:{px(4)}px {px(8)}px; border-bottom:1px solid {BORDER_COLOR}; }}"
            f"QListWidget::item:selected {{ background:{ACCENT_PURPLE}; color:white; }}"
            f"QListWidget::item:hover:!selected {{ background:{BG_HOVER}; }}")
        saved_lay.addWidget(self.list_saved)

        # 삭제 버튼 행
        del_row = QHBoxLayout(); del_row.setSpacing(5)
        self.btn_del_record = QPushButton("🗑  Delete Selected")
        self.btn_del_record.setFixedHeight(px(26))
        self.btn_del_record.setEnabled(False)
        self.btn_del_record.setStyleSheet(
            f"QPushButton {{ background:{ACCENT_RED}22; border:1px solid {ACCENT_RED}55;"
            f"  color:{ACCENT_RED}; border-radius:{px(5)}px; font-weight:700; font-size:{px(10)}px; }}"
            f"QPushButton:hover {{ background:{ACCENT_RED}44; }}"
            f"QPushButton:disabled {{ background:#2A2A3A; color:#444; border-color:#333; }}")
        self.btn_del_record.clicked.connect(self._delete_selected_record)
        del_row.addWidget(self.btn_del_record)

        self.btn_clear_records = QPushButton("✕  Clear All")
        self.btn_clear_records.setFixedHeight(px(26))
        self.btn_clear_records.setEnabled(False)
        self.btn_clear_records.setStyleSheet(
            f"QPushButton {{ background:{BG_CARD}; border:1px solid {BORDER_COLOR};"
            f"  color:{TEXT_SECONDARY}; border-radius:{px(5)}px; font-weight:700; font-size:{px(10)}px; }}"
            f"QPushButton:hover {{ background:{BG_HOVER}; color:{ACCENT_RED}; }}"
            f"QPushButton:disabled {{ background:#2A2A3A; color:#444; border-color:#333; }}")
        self.btn_clear_records.clicked.connect(self._clear_all_records)
        del_row.addWidget(self.btn_clear_records)
        saved_lay.addLayout(del_row)

        ctrl_lay.addWidget(grp_saved)
        self.list_saved.itemSelectionChanged.connect(
            lambda: self.btn_del_record.setEnabled(
                len(self.list_saved.selectedItems()) > 0))

        # ═══════════════════════════════════════════════════════
        #  📤 Export Excel 버튼  —  저장된 모든 레코드를 xlsx로 출력
        # ═══════════════════════════════════════════════════════
        self.btn_export = QPushButton("📤  Export All → Excel")
        self.btn_export.setMinimumHeight(px(38))
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet(
            f"QPushButton {{"
            f"  background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"    stop:0 #0A6E5E, stop:1 {ACCENT_GREEN});"
            f"  color:white; border:none; border-radius:{px(8)}px;"
            f"  font-size:{px(12)}px; font-weight:800; letter-spacing:1px; }}"
            f"QPushButton:hover {{"
            f"  background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"    stop:0 {ACCENT_GREEN}, stop:1 #06D6A0); }}"
            f"QPushButton:disabled {{ background:#2A2A3A; color:#555566; }}")
        self.btn_export.clicked.connect(self._export_excel)
        ctrl_lay.addWidget(self.btn_export)
        ctrl_lay.addStretch()

        # 스크롤 래핑
        scroll = QScrollArea()
        scroll.setWidget(ctrl_w)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(px(440))
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea {{ border:none; background:{BG_PANEL}; }}")

        # ── 우측 캔버스 ──────────────────────────────────────
        right_w = QWidget()
        right_lay = QVBoxLayout(right_w)
        right_lay.setContentsMargins(6, 6, 6, 6)
        right_lay.setSpacing(4)

        self.canvas = DVACanvas()
        nav = NavigationToolbar(self.canvas, right_w)
        nav.setStyleSheet(
            f"background:{BG_PANEL}; color:{TEXT_PRIMARY};"
            f" border-bottom:1px solid {BORDER_COLOR};")
        right_lay.addWidget(nav)
        right_lay.addWidget(self.canvas, stretch=1)

        spl = QSplitter(Qt.Horizontal)
        spl.addWidget(scroll)
        spl.addWidget(right_w)
        spl.setStretchFactor(0, 0)
        spl.setStretchFactor(1, 1)
        spl.setSizes([440, 860])
        root.addWidget(spl)

    # ── 디바운스 ─────────────────────────────────────────────
    def _schedule(self):
        """파일/사이클/smoothing/interp 변경 시 전체 재계산 (300ms debounce).
        _vq_timer가 먼저 fire하지 않도록 정지."""
        self._vq_timer.stop()        # 이중 렌더 방지
        self._est_cache = None       # estimated 캐시 무효화
        self._est_cache_key = None
        self._timer.start(300)

    def _schedule_vq(self):
        """mass/slip 파라미터만 변경 시 빠른 갱신 (80ms debounce).
        full recompute가 이미 예약돼 있으면 그쪽에 맡김."""
        if self._timer.isActive():
            return   # 전체 재계산이 예약됨 → 별도 갱신 불필요
        self._est_cache = None       # 파라미터 변경 → estimated 캐시 무효화
        self._est_cache_key = None
        self._vq_timer.start(80)

    # ── estimated 결과 캐싱 (파라미터 미변경 시 재계산 방지) ──
    def _compute_estimated_cached(self):
        """현재 spinbox 파라미터에 해당하는 estimated를 캐시 기반으로 반환."""
        key = (
            self.sp_pos_mass.value(), self.sp_neg_mass.value(),
            self.sp_pos_slip.value(), self.sp_neg_slip.value(),
            self.sp_smooth.value(),   self.sp_interp.value(),
            id(self._pos_seg_cache),  id(self._neg_seg_cache),
        )
        if self._est_cache is not None and self._est_cache_key == key:
            return self._est_cache
        est = dva_compute_estimated(
            self._pos_seg_cache, self._neg_seg_cache,
            pos_mass=self.sp_pos_mass.value(),
            neg_mass=self.sp_neg_mass.value(),
            pos_slip=self.sp_pos_slip.value(),
            neg_slip=self.sp_neg_slip.value(),
            smooth_w=self.sp_smooth.value(),
            interp_n=self.sp_interp.value(),
        )
        self._est_cache = est
        self._est_cache_key = key
        return est

    # ── 캐시 기반 전체 3패널 즉시 갱신 (mass/slip 변경 시 호출) ──
    def _refresh_all(self):
        """
        세그먼트 재로드 없이 캐시된 full_r / pos_seg / neg_seg 로
        현재 spinbox 파라미터를 반영해 3패널(dV/dQ + V-Q + Ref) 모두 재렌더.
        _recompute보다 훨씬 빠르고, 되돌아오는 버그도 없음.
        """
        if self._full_r_cache is None:
            return
        fl, pl, nl = self._labels_cache
        fit_range = (self.sp_fit_qlo.value(), self.sp_fit_qhi.value())

        # 캐시 기반 estimated 계산 (파라미터 동일하면 재계산 스킵)
        est = self._compute_estimated_cached()

        # 3패널 전체 render (mass/slip 변환 포함)
        self.canvas.render(
            self._full_r_cache, est, self._pos_seg_cache, self._neg_seg_cache,
            full_label=fl,
            pos_label=pl,
            neg_label=nl,
            fit_range=fit_range,
            pos_mass=self.sp_pos_mass.value(),
            pos_slip=self.sp_pos_slip.value(),
            neg_mass=self.sp_neg_mass.value(),
            neg_slip=self.sp_neg_slip.value(),
            records=self._saved_records,
            show_full_dvdq=self.chk_show_full_dvdq.isChecked(),
        )

    # ── 재계산 ───────────────────────────────────────────────
    def _recompute(self):
        full_seg, fc, fd = self.card_full.get_segment()
        pos_seg,  pc, pd = self.card_pos.get_segment()
        neg_seg,  nc, nd = self.card_neg.get_segment()

        # 파일 미로드
        if any(c.df is None for c in (self.card_full, self.card_pos, self.card_neg)):
            self.lbl_status.setText(
                "⏳  Full Cell / Positive Ref. / Negative Ref."
                " files to start automatic analysis.")
            return

        # 세그먼트 누락 확인
        missing = []
        if full_seg is None:
            missing.append(f"Full Cell — Cyc {fc} {fd}")
        if pos_seg is None:
            missing.append(f"Pos. Ref. — Cyc {pc} {pd}")
        if neg_seg is None:
            missing.append(f"Neg. Ref. — Cyc {nc} {nd}")
        if missing:
            self.lbl_status.setText(
                "⚠  No segment found:\n" + "\n".join(f"  · {m}" for m in missing))
            return

        # Full Cell dV/dQ
        params = dict(smooth_w=self.sp_smooth.value(),
                      interp_n=self.sp_interp.value(), dvdq_clip=50.0, peak_prom=0.3)
        full_r = compute_dvdq(full_seg, **params)
        if full_r is None:
            self.lbl_status.setText("⚠  Full Cell dV/dQ calculation failed")
            return

        # Estimated dV/dQ  —  Dahn JES 2012 Eq.[1]
        # dV/dQ(Q) = (1/mp)·dVp/dqp  −  (1/mn)·dVn/dqn
        # 세그먼트 캐시 (fitting + vq 갱신에서 재사용) — estimated 계산 전에 설정
        self._full_r_cache  = full_r
        self._pos_seg_cache = pos_seg
        self._neg_seg_cache = neg_seg
        self._est_cache = None       # 세그먼트 변경 → estimated 캐시 무효화
        self._est_cache_key = None

        est = self._compute_estimated_cached()

        # 상태 표시
        fd_s = 'Chg↑' if fd == 'charge' else 'Dis↓'
        pd_s = 'Chg↑' if pd == 'charge' else 'Dis↓'
        nd_s = 'Chg↑' if nd == 'charge' else 'Dis↓'

        # Measured vs Estimated 잔차 지표 계산
        metric_str = ""
        if est is not None:
            try:
                f_est = interp1d(est['Q_grid'], est['dvdq_estimated'],
                                 kind='linear', bounds_error=False, fill_value=np.nan)
                dv_est_on_meas = f_est(full_r['Q_i'])
                dv_meas = full_r['dvdq_s']
                ok = np.isfinite(dv_est_on_meas) & (dv_meas != 0)
                if ok.sum() > 5:
                    res = dv_est_on_meas[ok] - dv_meas[ok]
                    rms  = float(np.sqrt(np.mean(res**2)))
                    mae  = float(np.mean(np.abs(res)))
                    maxe = float(np.max(np.abs(res)))
                    ss_res = float(np.sum(res**2))
                    ss_tot = float(np.sum((dv_meas[ok] - np.mean(dv_meas[ok]))**2))
                    r2   = 1.0 - ss_res/ss_tot if ss_tot > 0 else float('nan')
                    r2_col = '#06D6A0' if r2 > 0.99 else '#F4A261' if r2 > 0.95 else '#EF233C'
                    r2_s   = f"{r2:.5f}" if np.isfinite(r2) else "N/A"
                    metric_str = (
                        f"<br><span style='color:{TEXT_SECONDARY};font-size:10px;'>"
                        f"RMS&nbsp;=&nbsp;<b style='color:white'>{rms:.6f}</b>&nbsp;V/mAh &nbsp;"
                        f"MAE&nbsp;=&nbsp;<b style='color:white'>{mae:.6f}</b>&nbsp;V/mAh<br>"
                        f"Max&nbsp;err&nbsp;=&nbsp;<b style='color:white'>{maxe:.6f}</b>&nbsp;V/mAh &nbsp;"
                        f"R²&nbsp;=&nbsp;<b style='color:{r2_col}'>{r2_s}</b></span>"
                    )
            except Exception:
                pass

        e_str = (f"Estimated Q: {est['Q_grid'].min():.2f}"
                 f"–{est['Q_grid'].max():.2f} mAh"
                 if est else "⚠ Estimated calculation failed")
        self.lbl_status.setText(
            f"<b style='color:{ACCENT_BLUE}'>Full</b>"
            f" Cyc {fc} {fd_s} | Q={full_r['Q_total']:.2f} mAh<br>"
            f"<b style='color:#F4A261'>Pos</b>"
            f" Cyc {pc} {pd_s} | m={self.sp_pos_mass.value():.4f}"
            f"  δ={self.sp_pos_slip.value():+.3f} mAh<br>"
            f"<b style='color:#06D6A0'>Neg</b>"
            f" Cyc {nc} {nd_s} | m={self.sp_neg_mass.value():.4f}"
            f"  δ={self.sp_neg_slip.value():+.3f} mAh"
            f"{metric_str}")
        self.lbl_status.setTextFormat(1)

        # 레이블 캐시 갱신
        self._labels_cache  = (
            f"Full Cell (Cyc {fc})",
            f"Pos. Ref. (Cyc {pc})",
            f"Neg. Ref. (Cyc {nc})",
        )

        # Fitting 범위 (Q축 범위가 설정됐으면 auto-update)
        q_total = full_r['Q_total']
        cur_lo = self.sp_fit_qlo.value()
        cur_hi = self.sp_fit_qhi.value()
        if cur_hi >= 9998:            # 초기값이면 자동 설정
            self.sp_fit_qlo.blockSignals(True)
            self.sp_fit_qhi.blockSignals(True)
            self.sp_fit_qlo.setValue(round(q_total * 0.05, 2))
            self.sp_fit_qhi.setValue(round(q_total * 0.95, 2))
            self.sp_fit_qlo.blockSignals(False)
            self.sp_fit_qhi.blockSignals(False)

        fit_range = (self.sp_fit_qlo.value(), self.sp_fit_qhi.value())

        # 렌더 — mass/slip 현재값을 전달해 V-Q 패널도 올바른 좌표로 그림
        self.canvas.render(
            full_r, est, pos_seg, neg_seg,
            full_label=self._labels_cache[0],
            pos_label=self._labels_cache[1],
            neg_label=self._labels_cache[2],
            fit_range=fit_range,
            pos_mass=self.sp_pos_mass.value(),
            pos_slip=self.sp_pos_slip.value(),
            neg_mass=self.sp_neg_mass.value(),
            neg_slip=self.sp_neg_slip.value(),
            records=self._saved_records,
            show_full_dvdq=self.chk_show_full_dvdq.isChecked(),
        )

    # ══════════════════════════════════════════════════════════
    #  저장 레코드 키 생성 헬퍼
    # ══════════════════════════════════════════════════════════
    @staticmethod
    def _make_record_key(fc, fd, pc, pd, nc, nd):
        """Full{fc}{fd}_Pos{pc}{pd}_Neg{nc}{nd} 형태의 고유 키 반환."""
        fd_s = 'Chg' if fd == 'charge' else 'Dis'
        pd_s = 'Chg' if pd == 'charge' else 'Dis'
        nd_s = 'Chg' if nd == 'charge' else 'Dis'
        return f"Full{fc}{fd_s}_Pos{pc}{pd_s}_Neg{nc}{nd_s}"

    # ══════════════════════════════════════════════════════════
    #  저장 목록 UI 갱신
    # ══════════════════════════════════════════════════════════
    def _refresh_saved_list(self):
        """_saved_records 딕셔너리를 QListWidget에 반영."""
        self.list_saved.clear()
        n = len(self._saved_records)
        for key, rec in self._saved_records.items():
            r2 = rec.get('r2', float('nan'))
            r2_s = f"{r2:.5f}" if np.isfinite(r2) else "N/A"
            mp = rec.get('mp', 0.0); mn = rec.get('mn', 0.0)
            dp = rec.get('dp', 0.0); dn = rec.get('dn', 0.0)
            item = QListWidgetItem(
                f"● {key}\n"
                f"   mp={mp:.4f}  mn={mn:.4f}  "
                f"δp={dp:+.3f}  δn={dn:+.3f}  R²={r2_s}")
            item.setData(Qt.UserRole, key)
            item.setForeground(QColor(TEXT_PRIMARY))
            self.list_saved.addItem(item)

        if n == 0:
            self.lbl_saved_count.setText("No saved records.")
        else:
            self.lbl_saved_count.setText(
                f"<b style='color:#F4A261'>{n}</b>"
                f"<span style='color:{TEXT_SECONDARY};'> record{'s' if n > 1 else ''} in memory</span>")
        self.lbl_saved_count.setTextFormat(1)

        has_data = n > 0
        self.btn_export.setEnabled(has_data)
        self.btn_clear_records.setEnabled(has_data)
        self.btn_del_record.setEnabled(
            has_data and len(self.list_saved.selectedItems()) > 0)

        # 파라미터 이력 그래프 갱신
        self.canvas.render_param_history(self._saved_records)

    # ══════════════════════════════════════════════════════════
    #  저장 목록 우클릭 컨텍스트 메뉴
    # ══════════════════════════════════════════════════════════
    def _saved_context_menu(self, pos):
        item = self.list_saved.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        act_del = menu.addAction("🗑  Delete this record")
        action = menu.exec_(self.list_saved.mapToGlobal(pos))
        if action == act_del:
            key = item.data(Qt.UserRole)
            self._saved_records.pop(key, None)
            self._refresh_saved_list()

    # ── 선택 항목 삭제 ────────────────────────────────────────
    def _delete_selected_record(self):
        selected = self.list_saved.selectedItems()
        if not selected:
            return
        key = selected[0].data(Qt.UserRole)
        reply = QMessageBox.question(
            self, "Delete Record",
            f"Delete record:\n  {key}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self._saved_records.pop(key, None)
            self._refresh_saved_list()

    # ── 전체 삭제 ─────────────────────────────────────────────
    def _clear_all_records(self):
        if not self._saved_records:
            return
        reply = QMessageBox.question(
            self, "Clear All",
            f"Delete all {len(self._saved_records)} saved record(s)?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self._saved_records.clear()
            self._refresh_saved_list()

    # ══════════════════════════════════════════════════════════
    #  💾 Save Current Fit  —  메모리에 임시 저장
    # ══════════════════════════════════════════════════════════
    def _save_current_fit(self):
        if self._full_r_cache is None:
            QMessageBox.warning(self, "No Data",
                "Please load all 3 files and complete the analysis first.")
            return

        full_seg, fc, fd = self.card_full.get_segment()
        pos_seg,  pc, pd = self.card_pos.get_segment()
        neg_seg,  nc, nd = self.card_neg.get_segment()

        if any(s is None for s in (full_seg, pos_seg, neg_seg)):
            QMessageBox.warning(self, "No Segment",
                "Could not find all required segments.\nCheck cycle numbers and direction.")
            return

        mp = self.sp_pos_mass.value()
        mn = self.sp_neg_mass.value()
        dp = self.sp_pos_slip.value()
        dn = self.sp_neg_slip.value()

        # 최신 파라미터로 estimated (캐시 활용)
        est = self._compute_estimated_cached()

        # 잔차 지표 계산 — fit range 내부 구간만 사용
        rms = mae = maxe = r2 = float('nan')
        q_lo = self.sp_fit_qlo.value()
        q_hi = self.sp_fit_qhi.value()
        if est is not None:
            try:
                f_est = interp1d(est['Q_grid'], est['dvdq_estimated'],
                                 kind='linear', bounds_error=False, fill_value=np.nan)
                Q_i    = self._full_r_cache['Q_i']
                dv_meas = self._full_r_cache['dvdq_s']
                # fit range 마스크 적용
                in_range = (Q_i >= q_lo) & (Q_i <= q_hi)
                dv_est_on_meas = f_est(Q_i)
                ok = in_range & np.isfinite(dv_est_on_meas) & (dv_meas != 0)
                if ok.sum() > 5:
                    res    = dv_est_on_meas[ok] - dv_meas[ok]
                    rms    = float(np.sqrt(np.mean(res ** 2)))
                    mae    = float(np.mean(np.abs(res)))
                    maxe   = float(np.max(np.abs(res)))
                    ss_res = float(np.sum(res ** 2))
                    ss_tot = float(np.sum((dv_meas[ok] - np.mean(dv_meas[ok])) ** 2))
                    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
            except Exception:
                pass

        record = dict(
            # 파라미터
            mp=mp, mn=mn, dp=dp, dn=dn,
            # 정밀도 지표
            rms=rms, mae=mae, maxe=maxe, r2=r2,
            # 처리 설정
            smooth_w=self.sp_smooth.value(),
            interp_n=self.sp_interp.value(),
            fit_q_lo=self.sp_fit_qlo.value(),
            fit_q_hi=self.sp_fit_qhi.value(),
            # 사이클 정보
            full_cyc=fc, full_dir=fd,
            pos_cyc=pc,  pos_dir=pd,
            neg_cyc=nc,  neg_dir=nd,
        )

        key = self._make_record_key(fc, fd, pc, pd, nc, nd)
        is_overwrite = key in self._saved_records
        self._saved_records[key] = record   # 같은 키면 덮어씀
        self._refresh_saved_list()

        r2_s = f"{r2:.5f}" if np.isfinite(r2) else "N/A"
        action_str = "Overwritten" if is_overwrite else "Saved"
        QMessageBox.information(
            self, f"Record {action_str}",
            f"{'⚠ Overwritten' if is_overwrite else '✓ Saved'}: {key}\n\n"
            f"mp = {mp:.5f}    mn = {mn:.5f}\n"
            f"δp = {dp:+.4f} mAh    δn = {dn:+.4f} mAh\n"
            f"R² = {r2_s}")

    # ══════════════════════════════════════════════════════════
    #  📤 Export All → Excel  —  저장된 모든 레코드를 xlsx로 출력
    #
    #  레이아웃: 하나의 시트(DVA Results)에 사이클별로 나란히 배치
    #    각 블록 = 8열
    #    행1    : 사이클 키 제목
    #    행2~11 : 파라미터 요약 (mp, mn, δp, δn, RMS, MAE, MaxErr, R², smooth, interp)
    #    행13~  : 헤더 + 데이터 (Measured Q/V/dVdQ | Estimated Q/dVdQ/pos/neg/V_full)
    # ══════════════════════════════════════════════════════════
    def _export_excel(self):
        if not self._saved_records:
            QMessageBox.warning(self, "No Records",
                "No saved records to export.\n"
                "Please save at least one fitting result first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export to Excel", "dva_results.xlsx",
            "Excel Files (*.xlsx);;All Files (*)")
        if not path:
            return

        try:
            import openpyxl
            from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
            from openpyxl.utils import get_column_letter
        except ImportError:
            QMessageBox.critical(self, "Missing Library",
                "openpyxl is not installed.\n\n"
                "Install with:  pip install openpyxl\n"
                "or:            conda install openpyxl")
            return

        try:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "DVA Results"

            # ── 스타일 헬퍼 ──────────────────────────────────
            def _fill(hex6):
                return PatternFill("solid", fgColor=hex6.lstrip('#'))

            def _font(bold=False, color="E8E8F0", size=10):
                return Font(bold=bold, color=color.lstrip('#'),
                            size=size, name="Segoe UI")

            def _border_thin():
                s = Side(style='thin', color="2A2A4A")
                return Border(left=s, right=s, top=s, bottom=s)

            def _align(h="left", v="center"):
                return Alignment(horizontal=h, vertical=v)

            FILL_TITLE  = _fill("#16162A")
            FILL_PARAM  = _fill("#1E1E35")
            FILL_HEAD   = _fill("#1A1A2E")
            FILL_ROW0   = _fill("#12122A")
            FILL_ROW1   = _fill("#0F0F1A")

            FONT_TITLE  = _font(bold=True,  color="4CC9F0", size=11)
            FONT_PLBL   = _font(bold=True,  color="F4A261", size=10)
            FONT_PVAL   = _font(bold=False, color="E8E8F0", size=10)
            FONT_HEAD   = _font(bold=True,  color="4CC9F0", size=9)
            FONT_DATA   = _font(bold=False, color="CCCCDD", size=9)

            BLOCK_W      = 9    # 각 사이클 블록 폭 (데이터 7열 + 구분 2열)
            PARAM_ROWS   = 10   # 파라미터 행 수
            DATA_HDR_ROW = PARAM_ROWS + 3   # 데이터 헤더 시작 행 (1-based)
            # 데이터 열 배치 (블록 내 상대 열 인덱스, 0-based):
            # 0:Q_meas  1:V_meas  2:dVdQ_meas | 3:Q_est  4:dVdQ_est  5:pos  6:neg  7:V_full

            col_start = 1   # 현재 블록 시작 열 (1-based)

            for key, rec in self._saved_records.items():
                c0 = col_start

                # ── 1행: 사이클 키 제목 ───────────────────────
                title_cell = ws.cell(row=1, column=c0, value=key)
                title_cell.font      = FONT_TITLE
                title_cell.fill      = FILL_TITLE
                title_cell.alignment = _align("center")
                title_cell.border    = _border_thin()
                # 7열에 걸쳐 병합
                ws.merge_cells(start_row=1, start_column=c0,
                               end_row=1,   end_column=c0 + 6)

                # ── 2~11행: 파라미터 요약 ─────────────────────
                params = [
                    ("mp  (Pos mass)",       rec.get('mp',       '')),
                    ("mn  (Neg mass)",       rec.get('mn',       '')),
                    ("δp / pos_slip (mAh)",  rec.get('dp',       '')),
                    ("δn / neg_slip (mAh)",  rec.get('dn',       '')),
                    ("RMS  (V/mAh)",         rec.get('rms',      '')),
                    ("MAE  (V/mAh)",         rec.get('mae',      '')),
                    ("Max Error  (V/mAh)",   rec.get('maxe',     '')),
                    ("R²",                   rec.get('r2',       '')),
                    ("Smooth window",        rec.get('smooth_w', '')),
                    ("Interp points",        rec.get('interp_n', '')),
                ]
                for i, (pname, pval) in enumerate(params):
                    row = 2 + i
                    lc = ws.cell(row=row, column=c0, value=pname)
                    lc.font = FONT_PLBL; lc.fill = FILL_PARAM
                    lc.alignment = _align(); lc.border = _border_thin()

                    if isinstance(pval, float):
                        display_val = round(pval, 8) if np.isfinite(pval) else "N/A"
                    else:
                        display_val = pval
                    vc = ws.cell(row=row, column=c0 + 1, value=display_val)
                    vc.font = FONT_PVAL; vc.fill = FILL_PARAM
                    vc.alignment = _align("right"); vc.border = _border_thin()

                # ── 데이터 헤더 행 ───────────────────────────
                data_headers = [
                    "Q_meas (mAh)", "V_meas (V)", "dVdQ_meas (V/mAh)",
                    "Q_est (mAh)",  "dVdQ_est",   "(1/mp)dVp", "(1/mn)dVn",
                    "V_full_est (V)",
                ]
                for j, lbl in enumerate(data_headers):
                    hc = ws.cell(row=DATA_HDR_ROW, column=c0 + j, value=lbl)
                    hc.font = FONT_HEAD; hc.fill = FILL_HEAD
                    hc.alignment = _align("center"); hc.border = _border_thin()

                # ── Measured 데이터 행 ───────────────────────
                # on-demand 재계산: 레코드에 저장된 파라미터로 full_r / est 재생성
                _fc = rec.get('full_cyc'); _fd = rec.get('full_dir')
                _pc = rec.get('pos_cyc');  _pd = rec.get('pos_dir')
                _nc = rec.get('neg_cyc');  _nd = rec.get('neg_dir')
                _full_seg = self.card_full.segs.get((_fc, _fd))
                _pos_seg  = self.card_pos.segs.get((_pc, _pd))
                _neg_seg  = self.card_neg.segs.get((_nc, _nd))

                full_r = {}
                if _full_seg is not None:
                    full_r = compute_dvdq(
                        _full_seg,
                        smooth_w=rec.get('smooth_w', 7),
                        interp_n=rec.get('interp_n', 1000),
                        dvdq_clip=50.0, peak_prom=0.3) or {}

                est = None
                if _pos_seg is not None and _neg_seg is not None:
                    est = dva_compute_estimated(
                        _pos_seg, _neg_seg,
                        pos_mass=rec.get('mp', 1.0),
                        neg_mass=rec.get('mn', 1.0),
                        pos_slip=rec.get('dp', 0.0),
                        neg_slip=rec.get('dn', 0.0),
                        smooth_w=rec.get('smooth_w', 7),
                        interp_n=rec.get('interp_n', 1000))

                Q_meas = full_r.get('Q_i',    np.array([]))
                V_meas = full_r.get('V_i',    np.array([]))
                dv_meas= full_r.get('dvdq_s', np.array([]))

                est = rec.get('est', None) if 'est' in rec else est  # 이전 버전 호환
                Q_est  = est.get('Q_grid',           np.array([])) if est else np.array([])
                dv_est = est.get('dvdq_estimated',   np.array([])) if est else np.array([])
                dv_pos = est.get('dvdq_pos_scaled',  np.array([])) if est else np.array([])
                dv_neg = est.get('dvdq_neg_scaled',  np.array([])) if est else np.array([])
                V_full = est.get('V_full_est',       np.array([])) if est else np.array([])

                # ── est 데이터를 Q_meas 축으로 보간 (Q축 불일치 해결) ──
                # 엑셀에서 같은 행이 같은 Q 위치를 가리키도록 통일
                if len(Q_est) > 1 and len(Q_meas) > 0:
                    def _interp_to_meas(y_src, x_src, x_dst):
                        if len(y_src) < 2: return np.full(len(x_dst), np.nan)
                        f = interp1d(x_src, y_src, kind='linear',
                                     bounds_error=False, fill_value=np.nan)
                        return f(x_dst)
                    dv_est_on_meas = _interp_to_meas(dv_est, Q_est, Q_meas)
                    dv_pos_on_meas = _interp_to_meas(dv_pos, Q_est, Q_meas)
                    dv_neg_on_meas = _interp_to_meas(dv_neg, Q_est, Q_meas)
                    V_full_on_meas = _interp_to_meas(V_full, Q_est, Q_meas)
                    # Q_est 열은 Q_meas와 동일하므로 Q_meas 사용
                    Q_est_out = Q_meas
                else:
                    dv_est_on_meas = dv_est
                    dv_pos_on_meas = dv_pos
                    dv_neg_on_meas = dv_neg
                    V_full_on_meas = V_full
                    Q_est_out      = Q_est

                n_rows = len(Q_meas)
                for ri in range(n_rows):
                    row  = DATA_HDR_ROW + 1 + ri
                    fill = FILL_ROW0 if ri % 2 == 0 else FILL_ROW1

                    # Measured 열 (0~2)
                    meas_vals = [
                        float(Q_meas[ri])  if ri < len(Q_meas)  else None,
                        float(V_meas[ri])  if ri < len(V_meas)  else None,
                        float(dv_meas[ri]) if ri < len(dv_meas) else None,
                    ]
                    # Estimated 열 (3~7) — Q_meas 축으로 보간된 값
                    est_vals = [
                        float(Q_est_out[ri])      if ri < len(Q_est_out)      else None,
                        float(dv_est_on_meas[ri]) if ri < len(dv_est_on_meas) and np.isfinite(dv_est_on_meas[ri]) else None,
                        float(dv_pos_on_meas[ri]) if ri < len(dv_pos_on_meas) and np.isfinite(dv_pos_on_meas[ri]) else None,
                        float(dv_neg_on_meas[ri]) if ri < len(dv_neg_on_meas) and np.isfinite(dv_neg_on_meas[ri]) else None,
                        float(V_full_on_meas[ri]) if (ri < len(V_full_on_meas) and
                                                       np.isfinite(V_full_on_meas[ri])) else None,
                    ]
                    for j, val in enumerate(meas_vals + est_vals):
                        dc = ws.cell(row=row, column=c0 + j,
                                     value=(round(val, 8) if val is not None else ""))
                        dc.font = FONT_DATA; dc.fill = fill
                        dc.alignment = _align("right"); dc.border = _border_thin()

                # ── 열 너비 설정 ─────────────────────────────
                col_widths = [14, 12, 18, 14, 14, 14, 14, 14]
                for j, w in enumerate(col_widths):
                    ws.column_dimensions[get_column_letter(c0 + j)].width = w
                # 구분 열 (블록 오른쪽 빈 열)
                ws.column_dimensions[get_column_letter(c0 + 8)].width = 2

                col_start += BLOCK_W

            # 행 높이
            ws.row_dimensions[1].height = 20
            for r in range(2, DATA_HDR_ROW + 1):
                ws.row_dimensions[r].height = 15
            # 스크롤 시 첫 행 고정
            ws.freeze_panes = "A2"

            wb.save(path)
            QMessageBox.information(
                self, "Export Complete",
                f"Successfully exported {len(self._saved_records)} "
                f"record(s) to:\n{path}")

        except Exception:
            import traceback
            QMessageBox.critical(self, "Export Error",
                traceback.format_exc()[:600])

    # ── Fitting 실행 ─────────────────────────────────────────────
    def _run_fit(self):
        if self._full_r_cache is None:
            QMessageBox.warning(self, "No Data",
                "Please load all 3 files and check the graph first.")
            return
        if self._fit_thread is not None and self._fit_thread.isRunning():
            return   # 이미 실행 중

        # 현재 파라미터를 초기값으로
        p0 = [
            self.sp_pos_mass.value(),
            self.sp_neg_mass.value(),
            self.sp_pos_slip.value(),
            self.sp_neg_slip.value(),
        ]

        # 파라미터별 bounds (체크박스 off = 고정)
        lo_list, hi_list = [], []
        for sp, chk, lo, hi in [
            (self.sp_pos_mass, self.chk_fit_mp, 0.01,  5.0),
            (self.sp_neg_mass, self.chk_fit_mn, 0.01,  5.0),
            (self.sp_pos_slip, self.chk_fit_dp, -500., 500.),
            (self.sp_neg_slip, self.chk_fit_dn, -500., 500.),
        ]:
            v = sp.value()
            if chk.isChecked():
                lo_list.append(lo)
                hi_list.append(hi)
            else:
                lo_list.append(v - 1e-9)
                hi_list.append(v + 1e-9)

        q_lo = self.sp_fit_qlo.value()
        q_hi = self.sp_fit_qhi.value()
        if q_lo >= q_hi:
            QMessageBox.warning(self, "Range Error", "Q From must be less than Q To.")
            return

        # 전략 매핑
        strat_map  = {0: 'multistart', 1: 'trf'}
        strat_name = {0: 'Multi-start', 1: 'TRF'}
        strategy   = strat_map[self.cmb_strategy.currentIndex()]
        sname      = strat_name[self.cmb_strategy.currentIndex()]

        self.btn_fit.setEnabled(False)
        self.btn_fit.setText("⏳  Fitting…")
        self.lbl_fit_result.setText(
            f"⏳ <b>{sname}</b> running…<br>"
            f"<span style='color:{TEXT_SECONDARY};font-size:10px;'>"
            f"Peak weight: {'ON' if self.chk_peak_weight.isChecked() else 'OFF'}</span>")
        self.lbl_fit_result.setTextFormat(1)

        tol      = float(self.cmb_tol.currentText())
        max_iter = 500   # 고정값

        self._fit_thread = FitThread(
            self._full_r_cache,
            self._pos_seg_cache,
            self._neg_seg_cache,
            p0=p0,
            bounds=(lo_list, hi_list),
            q_lo=q_lo, q_hi=q_hi,
            smooth_w=self.sp_smooth.value(),
            interp_n=self.sp_interp.value(),
            tol=tol,
            max_iter=max_iter,
            strategy=strategy,
            use_peak_weight=self.chk_peak_weight.isChecked(),
            multistart_n=5,   # 고정값
        )
        self._fit_thread.done.connect(self._on_fit_done)
        self._fit_thread.failed.connect(self._on_fit_failed)
        self._fit_thread.progress.connect(self.lbl_fit_result.setText)
        self._fit_thread.start()

    def _on_fit_done(self, res):
        """피팅 완료 → 결과를 SpinBox에 반영 + 그래프 업데이트."""
        self.btn_fit.setEnabled(True)
        self.btn_fit.setText("▶  AUTO FIT")

        self.sp_pos_mass.setValue(res['mp'])
        self.sp_neg_mass.setValue(res['mn'])
        self.sp_pos_slip.setValue(res['dp'])
        self.sp_neg_slip.setValue(res['dn'])

        ok_str  = "✓ Converged" if res['success'] else "△ Not converged (reference only)"
        r2_str  = f"{res['r2']:.5f}" if np.isfinite(res['r2']) else "N/A"
        r2_col  = '#06D6A0' if res['r2'] > 0.99 else '#F4A261' if res['r2'] > 0.95 else '#EF233C'
        sdisp   = {'multistart':'Multi-start','trf':'TRF'}.get(
                    res.get('strategy',''), '')
        self.lbl_fit_result.setText(
            f"<b style='color:#06D6A0'>{ok_str}</b>"
            f"<span style='color:{TEXT_SECONDARY};font-size:10px;'>"
            f" &nbsp;|&nbsp; {res['n_iter']} eval"
            f"{f'  [{sdisp}]' if sdisp else ''}</span><br>"
            f"<b style='color:#F4A261'>mp</b> = {res['mp']:.5f} &nbsp;"
            f"<b style='color:#F4A261'>δp</b> = {res['dp']:+.4f} mAh<br>"
            f"<b style='color:#06D6A0'>mn</b> = {res['mn']:.5f} &nbsp;"
            f"<b style='color:#06D6A0'>δn</b> = {res['dn']:+.4f} mAh<br>"
            f"<span style='color:{TEXT_SECONDARY};font-size:10px;'>"
            f"RMS&nbsp;=&nbsp;<b style='color:white'>{res['rms']:.6f}</b> V/mAh &nbsp;"
            f"MAE&nbsp;=&nbsp;<b style='color:white'>{res['mae']:.6f}</b> V/mAh<br>"
            f"Max&nbsp;err&nbsp;=&nbsp;<b style='color:white'>{res['maxe']:.6f}</b> V/mAh &nbsp;"
            f"R²&nbsp;=&nbsp;<b style='color:{r2_col}'>{r2_str}</b></span>")
        self.lbl_fit_result.setTextFormat(1)
        self._schedule()

    def _on_fit_failed(self, msg):
        self.btn_fit.setEnabled(True)
        self.btn_fit.setText("▶  AUTO FIT")
        self.lbl_fit_result.setText(f"<b style='color:#EF233C'>✗ Error</b><br>"
                                    f"<small>{msg[:200]}</small>")
        self.lbl_fit_result.setTextFormat(1)
        QMessageBox.critical(self, "Fitting Error", msg[:500])



# ══════════════════════════════════════════════════════════════
#  DVA Fitting  —  개선된 고정밀 피팅 엔진  (v9)
#
#  A. Smoothing 사전계산  — SG 필터를 루프 밖 1회 실행, 캐시 활용
#  B. n_grid 동적 향상   — optimizer 내부 해상도를 측정 데이터와 일치
#  C. Multi-start        — δ 격자 스캔으로 최적 초기점 자동 선택
#  D. Peak-weighted      — |dV/dQ| 크기 비례 가중치로 피크 매칭 강화
#  E. DE → TRF 2-stage   — Differential Evolution(전역) → TRF(정밀)
# ══════════════════════════════════════════════════════════════

def _precompute_half_cell(seg_df, smooth_w=21, interp_n=2000):
    """
    반쪽전지 세그먼트 → 고밀도 그리드 사전계산 캐시.
    fitting 루프 안에서는 이 캐시로 단순 보간만 수행 → 속도 대폭 향상.

    처리 순서 (Dubarry & Anseán 2022 best-practice):
      ① 보간 → ② 스무딩(V 곡선) → ③ 미분
    """
    Q, V = _seg_normalize_Q(seg_df)
    if Q is None or len(Q) < 5:
        return None

    # ① 보간: 균일 고밀도 Q 그리드 위에 V(Q) 재구성
    Q_dense = np.linspace(Q.min(), Q.max(), interp_n)
    kind = 'linear' if len(Q) < 4 else 'cubic'
    V_dense = interp1d(Q, V, kind=kind, fill_value='extrapolate')(Q_dense)

    # ② 스무딩: 미분 전에 V 곡선에 Savitzky-Golay 적용 (루프 밖 1회)
    w = smooth_w if smooth_w % 2 == 1 else smooth_w + 1
    w = max(w, 5)
    if len(V_dense) >= w:
        V_smooth = savgol_filter(V_dense, window_length=w, polyorder=min(3, w - 1))
    else:
        V_smooth = V_dense.copy()

    # ③ 미분: 스무딩된 V 곡선을 수치 미분 (clip 없이 저장, _fast_dvdq_on_grid에서 처리)
    dVdq_sm = np.gradient(V_smooth, Q_dense)

    return dict(Q_dense=Q_dense, V_dense=V_dense,
                dVdq_sm=dVdq_sm, Q_span=float(Q_dense[-1] - Q_dense[0]))


def _fast_dvdq_on_grid(cache, mass, slip, Q_grid, dvdq_clip=50.0):
    """
    캐시 기반 고속 mass·dV/dQ 계산.
    Dahn 정의: Q_cell = Q_dense / mass + slip
               dV/dQ_cell = mass · dV/dq_half
    """
    if cache is None:
        return np.zeros_like(Q_grid)

    # mass 스케일 변환 후 clip
    dVdQ_cell = cache['dVdq_sm'] * mass
    dVdQ_cell = np.clip(dVdQ_cell, -dvdq_clip, dvdq_clip)

    # 전지 Q 축: Q_cell = Q_dense / mass + slip  (Dahn 정의)
    Q_cell = cache['Q_dense'] / mass + slip

    return interp1d(Q_cell, dVdQ_cell,
                    kind='linear', bounds_error=False, fill_value=0.0)(Q_grid)


def _fast_dva_estimated(pos_cache, neg_cache,
                        pos_mass, neg_mass, pos_slip, neg_slip,
                        Q_grid, dvdq_clip=50.0):
    """캐시 기반 고속 estimated dV/dQ = pos항 + |neg항|."""
    if pos_cache is None or neg_cache is None:
        return np.ones(len(Q_grid)) * 1e6
    dvdq_pos = _fast_dvdq_on_grid(pos_cache, pos_mass, pos_slip, Q_grid, dvdq_clip)
    dvdq_neg = _fast_dvdq_on_grid(neg_cache, neg_mass, neg_slip, Q_grid, dvdq_clip)
    return dvdq_pos + np.abs(dvdq_neg)


class FitThread(QThread):
    done      = pyqtSignal(dict)
    failed    = pyqtSignal(str)
    progress  = pyqtSignal(str)   # 실시간 진행 메시지

    def __init__(self, full_r, pos_seg, neg_seg,
                 p0, bounds, q_lo, q_hi,
                 smooth_w, interp_n, tol=1e-8, max_iter=500,
                 strategy='multistart', use_peak_weight=True, multistart_n=5):
        super().__init__()
        self.full_r          = full_r
        self.pos_seg         = pos_seg
        self.neg_seg         = neg_seg
        self.p0              = p0
        self.bounds          = bounds
        self.q_lo            = q_lo
        self.q_hi            = q_hi
        self.smooth_w        = smooth_w
        self.interp_n        = interp_n
        self.tol             = tol
        self.max_iter        = max_iter
        self.strategy        = strategy
        self.use_peak_weight = use_peak_weight
        self.multistart_n    = multistart_n

    def run(self):
        try:
            result = dva_fit(
                self.full_r, self.pos_seg, self.neg_seg,
                self.p0, self.bounds,
                self.q_lo, self.q_hi,
                self.smooth_w, self.interp_n,
                tol=self.tol, max_iter=self.max_iter,
                strategy=self.strategy,
                use_peak_weight=self.use_peak_weight,
                multistart_n=self.multistart_n,
                progress_cb=self.progress.emit,
            )
            self.done.emit(result)
        except Exception:
            import traceback
            self.failed.emit(traceback.format_exc())


def dva_fit(full_r, pos_seg, neg_seg,
            p0, bounds, q_lo, q_hi,
            smooth_w=11, interp_n=1000,
            tol=1e-8, max_iter=500,
            strategy='multistart',
            use_peak_weight=True,
            multistart_n=5,
            progress_cb=None):
    """
    개선된 DVA 피팅.

    strategy:
      'multistart' — (mp, mn, δp, δn) 격자 스캔으로 최적 초기점 선택 후 TRF ★ 권장
      'trf'        — 단일 시작점 TRF (빠름, 초기값이 좋을 때 사용)

    use_peak_weight:
      True  — |dV/dQ| 크기 비례 가중치 (피크 매칭 강화)
    """
    from scipy.optimize import least_squares

    # ── Measured 데이터 마스킹 ───────────────────────────────
    Q_meas  = full_r['Q_i']
    dv_meas = full_r['dvdq_s']
    mask    = (Q_meas >= q_lo) & (Q_meas <= q_hi)
    if mask.sum() < 10:
        raise ValueError(
            f"Insufficient data in fitting range ({mask.sum()} points). "
            "Please widen the Q range.")
    Q_fit  = Q_meas[mask]
    dv_fit = dv_meas[mask]

    # ── n_grid를 측정 데이터 해상도에 맞춤 ──────────────────
    n_grid_fit   = max(len(Q_fit), 1000)
    Q_grid_fit   = np.linspace(Q_fit.min(), Q_fit.max(), n_grid_fit)
    dv_meas_grid = interp1d(Q_fit, dv_fit, kind='linear',
                            bounds_error=False, fill_value=0.0)(Q_grid_fit)

    # ── 피크 가중치 벡터 ────────────────────────────────────
    if use_peak_weight:
        abs_dv = np.abs(dv_meas_grid)
        w_lo   = np.percentile(abs_dv, 10)
        w_hi   = np.percentile(abs_dv, 95)
        span   = w_hi - w_lo if w_hi > w_lo else 1.0
        w_grid = 1.0 + 4.0 * np.clip((abs_dv - w_lo) / span, 0.0, 1.0)
    else:
        w_grid = np.ones(n_grid_fit)

    # ── Smoothing 사전계산 (루프 밖 1회) ─────────────────────
    if progress_cb: progress_cb("⏳ Preprocessing reference electrodes…")
    pos_cache = _precompute_half_cell(pos_seg, smooth_w=smooth_w, interp_n=interp_n)
    neg_cache = _precompute_half_cell(neg_seg, smooth_w=smooth_w, interp_n=interp_n)
    if pos_cache is None or neg_cache is None:
        raise ValueError("Failed to precompute half-cell data. Check reference segments.")

    # ── 잔차 / 비용 함수 ────────────────────────────────────
    def _residuals(params):
        mp, mn, dp, dn = params
        dv_est = _fast_dva_estimated(
            pos_cache, neg_cache, mp, mn, dp, dn, Q_grid_fit)
        return (dv_est - dv_meas_grid) * w_grid

    def _cost(params):
        r = _residuals(params)
        return float(np.dot(r, r))

    lo_arr = np.array(bounds[0], dtype=float)
    hi_arr = np.array(bounds[1], dtype=float)

    # ── TRF 정밀 수렴 헬퍼 ──────────────────────────────────
    def _refine(start):
        res = least_squares(
            _residuals, start, bounds=bounds, method='trf',
            ftol=tol, xtol=tol, gtol=tol,
            max_nfev=max_iter, verbose=0)
        return res.x, _cost(res.x), res.nfev, res.success

    # ═══════════════════════════════════════════════════════
    #  전략별 최적화
    # ═══════════════════════════════════════════════════════
    if strategy == 'trf':
        # 단일 시작점 TRF (기존 방식)
        if progress_cb: progress_cb("⏳ TRF optimization…")
        best_x, _, n_iter, success = _refine(p0)

    else:  # 'multistart'  ★ 기본 권장
        # ── (mp, mn, δp, δn) 4차원 격자 스캔 ────────────────
        # mass와 slip 모두 격자 탐색 (δ만 하던 기존보다 훨씬 넓게 탐색)
        if progress_cb: progress_cb("⏳ Multi-start: building grid…")

        N = multistart_n  # 각 축 분할 수

        # 탐색 범위: bounds 내에서, 현재 p0 주변을 중심으로
        # mass: p0 ± 0.3 범위 (0.7~1.3 정도), bounds 내로 클리핑
        # δ:    full_cell Q 범위의 ±50% 를 스캔 범위로 사용 (데이터 기반)
        q_span = float(Q_fit.max() - Q_fit.min())

        def _scan_range(lo, hi, center, half_span):
            lo_ = max(lo, center - half_span)
            hi_ = min(hi, center + half_span)
            if hi_ <= lo_: return np.array([center])
            return np.linspace(lo_, hi_, N)

        mp_range = _scan_range(lo_arr[0], hi_arr[0], p0[0], 0.4)
        mn_range = _scan_range(lo_arr[1], hi_arr[1], p0[1], 0.4)
        dp_range = _scan_range(lo_arr[2], hi_arr[2], p0[2], q_span * 0.5)
        dn_range = _scan_range(lo_arr[3], hi_arr[3], p0[3], q_span * 0.5)

        # 4차원 격자를 flat 배열로 생성 (NumPy meshgrid)
        _mp, _mn, _dp, _dn = np.meshgrid(
            mp_range, mn_range, dp_range, dn_range, indexing='ij')
        grid = np.column_stack([
            _mp.ravel(), _mn.ravel(), _dp.ravel(), _dn.ravel()])
        total = len(grid)

        if progress_cb: progress_cb(
            f"⏳ Grid scan  ({total} points)…")

        # 격자 스캔 — 진행률 보고를 위해 블록 단위로 분할
        best_cost  = np.inf
        best_start = list(p0)
        report_interval = max(1, total // 10)
        for i, trial in enumerate(grid):
            c = _cost(trial)
            if c < best_cost:
                best_cost  = c
                best_start = trial.tolist()
            if progress_cb and (i + 1) % report_interval == 0:
                progress_cb(
                    f"⏳ Grid scan  {i+1}/{total}  "
                    f"(best cost={best_cost:.5f})…")

        if progress_cb: progress_cb(
            f"⏳ TRF refinement from best start…")
        best_x, _, n_refine, success = _refine(best_start)
        n_iter = total + n_refine

    # ── 최종 오차 지표 (가중치 없이 순수 잔차) ───────────────
    mp_opt, mn_opt, dp_opt, dn_opt = best_x
    dv_est_final = _fast_dva_estimated(
        pos_cache, neg_cache, mp_opt, mn_opt, dp_opt, dn_opt, Q_grid_fit)
    res_final = dv_est_final - dv_meas_grid
    rms  = float(np.sqrt(np.mean(res_final ** 2)))
    mae  = float(np.mean(np.abs(res_final)))
    maxe = float(np.max(np.abs(res_final)))
    ss_res = float(np.sum(res_final ** 2))
    ss_tot = float(np.sum((dv_meas_grid - np.mean(dv_meas_grid)) ** 2))
    r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    # 시각화용 estimated 재계산 (표준 경로 사용)
    if progress_cb: progress_cb("⏳ Finalizing visualization…")
    est_opt = dva_compute_estimated(
        pos_seg, neg_seg,
        pos_mass=mp_opt, neg_mass=mn_opt,
        pos_slip=dp_opt, neg_slip=dn_opt,
        smooth_w=smooth_w, interp_n=interp_n,
    )

    return dict(
        mp=mp_opt, mn=mn_opt,
        dp=dp_opt, dn=dn_opt,
        rms=rms, mae=mae, maxe=maxe, r2=r2,
        n_iter=n_iter,
        success=success,
        strategy=strategy,
        est=est_opt,
    )


# ══════════════════════════════════════════════════════════════
#  "DVA" 텍스트 아이콘 생성
# ══════════════════════════════════════════════════════════════
def _make_dva_icon(size=64):
    """QPainter로 'DVA' 텍스트가 새겨진 QIcon 생성."""
    px = QPixmap(size, size)
    px.fill(Qt.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)

    # 배경 원
    grad = QLinearGradient(0, 0, size, size)
    grad.setColorAt(0.0, QColor("#7B2FBE"))
    grad.setColorAt(1.0, QColor("#4CC9F0"))
    p.setBrush(QBrush(grad))
    p.setPen(Qt.NoPen)
    p.drawEllipse(2, 2, size - 4, size - 4)

    # "DVA" 텍스트
    font = QFont("Segoe UI", int(size * 0.30), QFont.Black)
    font.setLetterSpacing(QFont.AbsoluteSpacing, 1.5)
    p.setFont(font)
    p.setPen(QColor("white"))
    p.drawText(px.rect(), Qt.AlignCenter, "DVA")
    p.end()
    return QIcon(px)


# ══════════════════════════════════════════════════════════════
#  메인 윈도우  (DVA 탭만)
# ══════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DVA Tool_v16")
        self.setWindowIcon(_make_dva_icon())
        self.resize(px(1440), px(860))
        self.setMinimumSize(px(1000), px(620))
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.dva_panel = DVAPanel()
        root.addWidget(self.dva_panel)
        sb = QStatusBar()
        self.setStatusBar(sb)
        sb.showMessage(
            "  Ready — Please load Full Cell / Positive Ref. / Negative Ref. files")
        # 제작자 표시 (우측 하단 고정)
        author_lbl = QLabel("Kwonhyung Lee  ")
        author_lbl.setStyleSheet(
            f"color:#C8C8E0; font-size:{px(10)}px; font-style:italic;")
        sb.addPermanentWidget(author_lbl)


# ══════════════════════════════════════════════════════════════
#  스플래시 스크린
# ══════════════════════════════════════════════════════════════
def _make_splash():
    """로딩 중 표시할 스플래시 픽스맵 생성."""
    _s = px   # 전역 px() 헬퍼 (스플래시 내부에서 QPixmap 변수명과 충돌 방지)
    W, H = _s(480), _s(280)
    pixmap = QPixmap(W, H)
    pixmap.fill(Qt.transparent)

    p = QPainter(pixmap)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)

    # ── 배경 라운드 박스 ──────────────────────────────────────
    path = QPainterPath()
    path.addRoundedRect(0, 0, W, H, _s(18), _s(18))
    grad = QLinearGradient(0, 0, 0, H)
    grad.setColorAt(0.0, QColor("#0D0D1F"))
    grad.setColorAt(1.0, QColor("#12122A"))
    p.fillPath(path, QBrush(grad))

    # 테두리
    p.setPen(QPen(QColor("#4CC9F055"), 2))
    p.drawPath(path)

    # ── DVA 아이콘 (좌측) ─────────────────────────────────────
    icon_size = _s(64)
    grad_icon = QLinearGradient(_s(20), _s(30), _s(20) + icon_size, _s(30) + icon_size)
    grad_icon.setColorAt(0.0, QColor("#7B2FBE"))
    grad_icon.setColorAt(1.0, QColor("#4CC9F0"))
    p.setBrush(QBrush(grad_icon))
    p.setPen(Qt.NoPen)
    p.drawEllipse(_s(28), _s(28), icon_size, icon_size)

    font_icon = QFont("Segoe UI", _s(18), QFont.Black)
    p.setFont(font_icon)
    p.setPen(QColor("white"))
    p.drawText(_s(28), _s(28), icon_size, icon_size, Qt.AlignCenter, "DVA")

    # ── 타이틀 ───────────────────────────────────────────────
    font_title = QFont("Segoe UI", _s(20), QFont.Bold)
    p.setFont(font_title)
    grad_text = QLinearGradient(_s(110), 0, _s(460), 0)
    grad_text.setColorAt(0.0, QColor("#4CC9F0"))
    grad_text.setColorAt(1.0, QColor("#7B2FBE"))
    p.setPen(QPen(QColor("#4CC9F0")))
    p.drawText(_s(108), _s(35), _s(350), _s(55), Qt.AlignLeft | Qt.AlignVCenter,
               "DVA Tool_v16")

    # ── 구분선 ───────────────────────────────────────────────
    p.setPen(QPen(QColor("#4CC9F033"), 1))
    p.drawLine(_s(28), _s(110), W - _s(28), _s(110))

    # ── 로딩 메시지 ──────────────────────────────────────────
    font_msg = QFont("Segoe UI", _s(10))
    p.setFont(font_msg)
    p.setPen(QColor("#AAAACC"))
    p.drawText(0, _s(120), W, _s(30), Qt.AlignCenter, "Initializing modules…")

    # ── 프로그레스 바 (정적) ──────────────────────────────────
    bar_x, bar_y, bar_w, bar_h = _s(40), _s(168), W - _s(80), _s(8)
    # 배경
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#1E1E3A"))
    p.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, _s(4), _s(4))
    # 진행 (애니메이션 효과를 주기 위해 그라데이션으로 채움)
    grad_bar = QLinearGradient(bar_x, 0, bar_x + bar_w, 0)
    grad_bar.setColorAt(0.0, QColor("#7B2FBE"))
    grad_bar.setColorAt(0.5, QColor("#4CC9F0"))
    grad_bar.setColorAt(1.0, QColor("#7B2FBE88"))
    p.setBrush(QBrush(grad_bar))
    p.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, _s(4), _s(4))

    # ── 점 애니메이터 표시 ───────────────────────────────────
    dot_y = _s(196)
    for i, (cx, alpha) in enumerate([(W//2 - _s(24), 255),
                                      (W//2,           180),
                                      (W//2 + _s(24),  100)]):
        c = QColor("#4CC9F0")
        c.setAlpha(alpha)
        p.setBrush(c)
        p.setPen(Qt.NoPen)
        p.drawEllipse(cx - _s(5), dot_y - _s(5), _s(10), _s(10))

    # ── 하단 제작자 ──────────────────────────────────────────
    font_auth = QFont("Segoe UI", _s(9))
    font_auth.setItalic(True)
    p.setFont(font_auth)
    p.setPen(QColor("#555577"))
    p.drawText(0, _s(230), W, _s(30), Qt.AlignCenter, "Kwonhyung Lee")

    p.end()

    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
    splash.setMask(pixmap.mask())
    return splash


# ══════════════════════════════════════════════════════════════
#  진입점
# ══════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("DVA Tool_v16")
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)
    app.setWindowIcon(_make_dva_icon())

    # ── 해상도 감지 → UI_SCALE 설정 ───────────────────────────
    global UI_SCALE
    screen = app.primaryScreen()
    screen_w = screen.size().width()
    screen_h = screen.size().height()
    # 가로·세로 비율 중 작은 값을 사용 (UI가 잘리지 않도록)
    scale_w = screen_w / _BASE_W
    scale_h = screen_h / _BASE_H
    UI_SCALE = round(min(scale_w, scale_h), 4)

    # ── 스케일된 스타일시트 전역 적용 ────────────────────────
    app.setStyleSheet(_scaled_stylesheet())

    # ── 스플래시 표시 ─────────────────────────────────────────
    splash = _make_splash()
    splash.show()
    app.processEvents()   # 즉시 렌더링

    # ── 메인 윈도우 생성 (무거운 import/init 여기서 발생) ──────
    win = MainWindow()

    # ── 스플래시 닫고 메인 윈도우 표시 ───────────────────────
    win.show()
    splash.finish(win)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
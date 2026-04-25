
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Hisse Analiz Paneli", layout="wide")

st.title("📈 Hisse Analiz Paneli")
st.caption("Teknik analiz, trend okuma, destek/direnç, hedef bölgeler ve otomatik yorumlama paneli.")

DISCLAIMER = """
⚠️ Bu uygulama yatırım tavsiyesi vermez. Üretilen sinyaller teknik göstergelere dayalı otomatik yorumdur.
Al/sat/elde tut kararını tek başına bu panele göre vermeyin.
"""
st.warning(DISCLAIMER)

# -----------------------------
# Helpers
# -----------------------------

def normalize_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if not s:
        return s
    # BIST kolaylığı: THYAO yazılırsa THYAO.IS yap
    if "." not in s and not s.startswith("^") and "-" not in s:
        return s + ".IS"
    return s

@st.cache_data(ttl=900)
def load_price_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.dropna()
    return df

@st.cache_data(ttl=3600)
def load_info(symbol: str):
    t = yf.Ticker(symbol)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}
    holders = None
    try:
        holders = t.institutional_holders
    except Exception:
        holders = None
    return info, holders

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]

    out["SMA20"] = close.rolling(20).mean()
    out["SMA50"] = close.rolling(50).mean()
    out["SMA200"] = close.rolling(200).mean()
    out["EMA12"] = close.ewm(span=12, adjust=False).mean()
    out["EMA26"] = close.ewm(span=26, adjust=False).mean()

    out["MACD"] = out["EMA12"] - out["EMA26"]
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))

    out["BB_MID"] = close.rolling(20).mean()
    out["BB_STD"] = close.rolling(20).std()
    out["BB_UP"] = out["BB_MID"] + 2 * out["BB_STD"]
    out["BB_LOW"] = out["BB_MID"] - 2 * out["BB_STD"]

    high_low = out["High"] - out["Low"]
    high_close = (out["High"] - out["Close"].shift()).abs()
    low_close = (out["Low"] - out["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["ATR14"] = tr.rolling(14).mean()

    out["VOL20"] = out["Volume"].rolling(20).mean()
    return out

def nearest_levels(df: pd.DataFrame, window: int = 80):
    d = df.tail(window).copy()
    if d.empty:
        return None
    last = float(d["Close"].iloc[-1])
    lows = d["Low"].rolling(5, center=True).min()
    highs = d["High"].rolling(5, center=True).max()

    support_candidates = d.loc[d["Low"].eq(lows), "Low"].dropna().values
    resist_candidates = d.loc[d["High"].eq(highs), "High"].dropna().values

    supports = sorted([x for x in support_candidates if x < last], reverse=True)[:3]
    resistances = sorted([x for x in resist_candidates if x > last])[:3]

    if len(supports) == 0:
        supports = [float(d["Low"].min())]
    if len(resistances) == 0:
        resistances = [float(d["High"].max())]

    return last, supports, resistances

def fib_levels(df: pd.DataFrame, window: int = 120):
    d = df.tail(window)
    if d.empty:
        return {}
    high = float(d["High"].max())
    low = float(d["Low"].min())
    diff = high - low
    return {
        "0.0% Dip": low,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50.0%": high - 0.5 * diff,
        "61.8%": high - 0.618 * diff,
        "78.6%": high - 0.786 * diff,
        "100% Tepe": high,
    }

def trend_score(df: pd.DataFrame):
    latest = df.iloc[-1]
    score = 0
    notes = []

    close = latest["Close"]
    rsi = latest.get("RSI14", np.nan)
    macd = latest.get("MACD", np.nan)
    macd_signal = latest.get("MACD_SIGNAL", np.nan)

    if close > latest.get("SMA20", np.nan): score += 1; notes.append("Fiyat SMA20 üzerinde: kısa vadeli yapı pozitif.")
    else: score -= 1; notes.append("Fiyat SMA20 altında: kısa vadede baskı var.")

    if close > latest.get("SMA50", np.nan): score += 1; notes.append("Fiyat SMA50 üzerinde: orta vadeli trend destekleniyor.")
    else: score -= 1; notes.append("Fiyat SMA50 altında: orta vadede zayıflık var.")

    if close > latest.get("SMA200", np.nan): score += 2; notes.append("Fiyat SMA200 üzerinde: ana trend pozitif.")
    else: score -= 2; notes.append("Fiyat SMA200 altında: ana trend zayıf.")

    if macd > macd_signal: score += 1; notes.append("MACD sinyal üstünde: momentum pozitif.")
    else: score -= 1; notes.append("MACD sinyal altında: momentum zayıf.")

    if pd.notna(rsi):
        if rsi < 30:
            score += 1
            notes.append("RSI 30 altına yakın/altında: aşırı satım bölgesi izlenebilir.")
        elif rsi > 70:
            score -= 1
            notes.append("RSI 70 üstünde: aşırı alım ve düzeltme riski var.")
        elif 45 <= rsi <= 60:
            score += 0.5
            notes.append("RSI dengeli bölgede: trend teyidi için kırılım beklenebilir.")
        else:
            notes.append("RSI nötr bölgede.")

    if score >= 3:
        label = "Pozitif / Güçlü trend"
    elif score >= 1:
        label = "Nötr-pozitif / İzlenebilir"
    elif score > -2:
        label = "Kararsız / Yatay-riskli"
    else:
        label = "Negatif / Zayıf trend"

    return score, label, notes

def make_chart(df: pd.DataFrame, symbol: str, show_bb=True):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.52, 0.16, 0.16, 0.16],
        vertical_spacing=0.03,
        subplot_titles=("Fiyat", "Hacim", "RSI", "MACD")
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Mum"
    ), row=1, col=1)

    for col in ["SMA20", "SMA50", "SMA200"]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col), row=1, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_UP"], mode="lines", name="BB Üst", line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOW"], mode="lines", name="BB Alt", line=dict(width=1)), row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Hacim"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["VOL20"], mode="lines", name="Hacim Ort."), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], mode="lines", name="RSI14"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], mode="lines", name="MACD Sinyal"), row=4, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="MACD Hist"), row=4, col=1)

    fig.update_layout(
        height=950,
        xaxis_rangeslider_visible=False,
        title=f"{symbol} Teknik Grafik",
        legend_orientation="h",
    )
    return fig

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("Ayarlar")
raw_symbol = st.sidebar.text_input("Hisse / Sembol", value="THYAO")
symbol = normalize_symbol(raw_symbol)

period = st.sidebar.selectbox("Veri dönemi", ["6mo", "1y", "2y", "5y", "10y"], index=2)
interval = st.sidebar.selectbox("Periyot", ["1d", "1wk", "1mo"], index=0)
show_bb = st.sidebar.checkbox("Bollinger Bantları göster", value=True)

st.sidebar.markdown("""
**BIST örnekleri**
- THYAO → THYAO.IS
- ISCTR → ISCTR.IS
- TUPRS → TUPRS.IS
- EREGL → EREGL.IS
""")

# -----------------------------
# Main
# -----------------------------

df = load_price_data(symbol, period, interval)

if df.empty:
    st.error("Veri bulunamadı. BIST için sembolü THYAO, ISCTR veya THYAO.IS gibi deneyin.")
    st.stop()

df = add_indicators(df).dropna()

if len(df) < 60:
    st.warning("Veri az geldi. Daha uzun dönem seçmek daha sağlıklı sonuç verir.")

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest
daily_change = (latest["Close"] / prev["Close"] - 1) * 100 if prev["Close"] else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Son Fiyat", f"{latest['Close']:.2f}", f"{daily_change:.2f}%")
c2.metric("RSI 14", f"{latest['RSI14']:.1f}")
c3.metric("MACD", f"{latest['MACD']:.2f}")
c4.metric("ATR 14", f"{latest['ATR14']:.2f}")
c5.metric("Hacim / Ort.", f"{latest['Volume'] / latest['VOL20']:.2f}x" if latest["VOL20"] else "-")

fig = make_chart(df, symbol, show_bb)
st.plotly_chart(fig, use_container_width=True)

score, label, notes = trend_score(df)
levels = nearest_levels(df)
fib = fib_levels(df)

st.subheader("🤖 Otomatik Grafik Yorumu")

col_a, col_b = st.columns([1, 1])
with col_a:
    st.markdown(f"### Genel sinyal: **{label}**")
    st.markdown(f"**Skor:** `{score}`")
    st.write("**Gerekçe:**")
    for n in notes:
        st.write("- " + n)

with col_b:
    if levels:
        last, supports, resistances = levels
        st.write("**Destek bölgeleri:**")
        for s in supports:
            st.write(f"- {s:.2f}  ({(s/last-1)*100:.2f}%)")
        st.write("**Direnç / hedef bölgeleri:**")
        for r in resistances:
            st.write(f"- {r:.2f}  ({(r/last-1)*100:.2f}%)")

st.subheader("🎯 Senaryo Bazlı Hedef / Risk")
if levels:
    last, supports, resistances = levels
    first_support = supports[0]
    first_resistance = resistances[0]
    atr = float(latest["ATR14"])

    table = pd.DataFrame({
        "Senaryo": ["Koruma / Stop bölgesi", "İlk direnç", "ATR bazlı kısa hedef", "ATR bazlı risk"],
        "Seviye": [
            round(first_support, 2),
            round(first_resistance, 2),
            round(last + 1.5 * atr, 2),
            round(last - 1.5 * atr, 2),
        ],
        "Son fiyata göre %": [
            round((first_support / last - 1) * 100, 2),
            round((first_resistance / last - 1) * 100, 2),
            round(((last + 1.5 * atr) / last - 1) * 100, 2),
            round(((last - 1.5 * atr) / last - 1) * 100, 2),
        ]
    })
    st.dataframe(table, use_container_width=True)

st.subheader("📐 Fibonacci Bölgeleri")
fib_df = pd.DataFrame({"Seviye": fib.keys(), "Fiyat": [round(v, 2) for v in fib.values()]})
st.dataframe(fib_df, use_container_width=True)

st.subheader("🏢 Şirket / Sahiplik Bilgileri")
info, holders = load_info(symbol)

info_cols = st.columns(4)
info_cols[0].metric("Piyasa Değeri", f"{info.get('marketCap', 0):,.0f}" if info.get("marketCap") else "-")
info_cols[1].metric("F/K", f"{info.get('trailingPE'):.2f}" if info.get("trailingPE") else "-")
info_cols[2].metric("PD/DD", f"{info.get('priceToBook'):.2f}" if info.get("priceToBook") else "-")
info_cols[3].metric("Beta", f"{info.get('beta'):.2f}" if info.get("beta") else "-")

with st.expander("Detaylı şirket bilgisi"):
    st.write("Şirket adı:", info.get("longName", "-"))
    st.write("Sektör:", info.get("sector", "-"))
    st.write("Endüstri:", info.get("industry", "-"))
    st.write("Web:", info.get("website", "-"))
    st.write("Özet:", info.get("longBusinessSummary", "-"))

if holders is not None and not holders.empty:
    st.write("Kurumsal sahiplik / holder tablosu:")
    st.dataframe(holders, use_container_width=True)
else:
    st.info("Bu sembol için Yahoo Finance üzerinde lot/sahiplik tablosu bulunamadı. BIST tarafında bu veri çoğu hissede ücretsiz kaynakta eksik olabilir.")

st.subheader("✅ Sonuç Özeti")
summary_lines = []
summary_lines.append(f"{symbol} için otomatik analiz sonucu: {label}.")
if latest["Close"] > latest["SMA50"] and latest["Close"] > latest["SMA200"]:
    summary_lines.append("Fiyat hem SMA50 hem SMA200 üzerinde olduğu için orta-uzun vadeli teknik görünüm destekleniyor.")
elif latest["Close"] < latest["SMA50"] and latest["Close"] < latest["SMA200"]:
    summary_lines.append("Fiyat SMA50 ve SMA200 altında olduğu için trend zayıf; tepki yükselişleri dirençlerde zorlanabilir.")
else:
    summary_lines.append("Fiyat ortalamalar arasında; net yön için kırılım beklemek daha sağlıklı.")

if latest["RSI14"] > 70:
    summary_lines.append("RSI aşırı alımda; yeni alım için geri çekilme veya yatay soğuma beklenebilir.")
elif latest["RSI14"] < 30:
    summary_lines.append("RSI aşırı satımda; tepki potansiyeli var ama trend teyidi şart.")
else:
    summary_lines.append("RSI nötr bölgede; karar için destek/direnç ve hacim teyidi önemli.")

if latest["Volume"] > latest["VOL20"] * 1.5:
    summary_lines.append("Hacim 20 günlük ortalamanın belirgin üzerinde; hareketin ciddiyeti artmış olabilir.")
else:
    summary_lines.append("Hacim olağan seviyede; güçlü kırılım için hacim artışı aranmalı.")

for line in summary_lines:
    st.write("- " + line)

st.download_button(
    "Analiz verisini CSV indir",
    data=df.to_csv().encode("utf-8"),
    file_name=f"{symbol}_analiz.csv",
    mime="text/csv"
)

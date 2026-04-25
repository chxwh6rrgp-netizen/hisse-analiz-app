
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="BIST PRO Hisse Analiz Paneli", layout="wide")

BIST_WATCHLIST = [
    "THYAO","TUPRS","EREGL","ASELS","KCHOL","SAHOL","SISE","BIMAS","AKBNK","GARAN",
    "ISCTR","YKBNK","PETKM","FROTO","TOASO","KRDMD","PGSUS","TCELL","KOZAL","ENKAI",
    "HEKTS","SASA","ASTOR","KONTR","ALARK","ODAS","PEKGY","GUBRF","TAVHL","ARCLK"
]

def normalize_symbol(s):
    s = str(s).upper().strip()
    if not s.endswith(".IS"):
        s += ".IS"
    return s

@st.cache_data(ttl=900)
def load_data(symbol, period="1y", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df, n=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    return line, signal, hist

def indicators(df):
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["RSI"] = rsi(df["Close"], 14)
    df["ATR"] = atr(df, 14)
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = macd(df["Close"])
    df["VOL_AVG20"] = df["Volume"].rolling(20).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_AVG20"]
    df["BB_MID"] = df["Close"].rolling(20).mean()
    df["BB_STD"] = df["Close"].rolling(20).std()
    df["BB_UP"] = df["BB_MID"] + 2 * df["BB_STD"]
    df["BB_LOW"] = df["BB_MID"] - 2 * df["BB_STD"]
    direction = np.sign(df["Close"].diff()).fillna(0)
    df["OBV"] = (direction * df["Volume"]).cumsum()
    return df.dropna()

def support_resistance(df, lookback=120):
    d = df.tail(lookback)
    support = float(d["Low"].rolling(5).min().dropna().quantile(0.20))
    resistance = float(d["High"].rolling(5).max().dropna().quantile(0.80))
    high = float(d["High"].max())
    low = float(d["Low"].min())
    return support, resistance, high, low

def score_stock(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    notes = []

    if last["Close"] > last["SMA20"] > last["SMA50"]:
        score += 20; notes.append("Kısa vadeli trend güçlü.")
    elif last["Close"] > last["SMA50"]:
        score += 12; notes.append("Fiyat 50 günlük ortalama üzerinde.")
    else:
        notes.append("Trend henüz güçlü değil.")

    if 45 <= last["RSI"] <= 65:
        score += 18; notes.append("RSI sağlıklı pozitif bölgede.")
    elif 35 <= last["RSI"] < 45:
        score += 12; notes.append("RSI dipten dönüş bölgesine yakın.")
    elif last["RSI"] > 70:
        score += 4; notes.append("RSI aşırı alım bölgesine yakın.")
    else:
        score += 6; notes.append("RSI zayıf/kararsız.")

    if last["MACD"] > last["MACD_SIGNAL"] and last["MACD_HIST"] > prev["MACD_HIST"]:
        score += 20; notes.append("MACD pozitif ve güçleniyor.")
    elif last["MACD"] > last["MACD_SIGNAL"]:
        score += 14; notes.append("MACD pozitif.")
    else:
        notes.append("MACD henüz net pozitif değil.")

    if last["VOL_RATIO"] >= 1.5:
        score += 18; notes.append("Hacim belirgin yüksek.")
    elif last["VOL_RATIO"] >= 1.0:
        score += 10; notes.append("Hacim ortalama üstü.")
    else:
        notes.append("Hacim desteği zayıf.")

    if last["Close"] <= last["BB_LOW"] * 1.04:
        score += 14; notes.append("Bollinger alt banda yakın: alım bölgesi olabilir.")
    elif last["Close"] < last["BB_MID"]:
        score += 8; notes.append("Fiyat orta bandın altında, takip edilebilir.")
    elif last["Close"] < last["BB_UP"]:
        score += 10; notes.append("Fiyat Bollinger içinde sağlıklı.")
    else:
        score += 2; notes.append("Üst banda yakın, kısa vadeli risk artmış.")

    if len(df) > 30 and df["OBV"].iloc[-1] > df["OBV"].iloc[-20]:
        score += 10; notes.append("OBV yükseliyor: para girişi işareti.")
    else:
        notes.append("OBV güçlü para girişi göstermiyor.")

    score = min(int(score), 100)

    if score >= 80:
        karar = "🔥 GÜÇLÜ AL / FIRSAT"
    elif score >= 65:
        karar = "🟢 ALIM BÖLGESİ / POZİTİF"
    elif score >= 50:
        karar = "🟡 TAKİP"
    else:
        karar = "🔴 ZAYIF / UZAK DUR"

    return score, karar, notes

def target_stop(df):
    last = float(df["Close"].iloc[-1])
    a = float(df["ATR"].iloc[-1])
    support, resistance, high, low = support_resistance(df)
    stop = min(support, last - 1.5 * a)
    target1 = max(resistance, last + 1.8 * a)
    target2 = last + 3.0 * a
    risk = max(last - stop, 0.01)
    reward = target1 - last
    rr = reward / risk if risk > 0 else np.nan
    return support, resistance, stop, target1, target2, rr

def plot_chart(df, symbol):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.52, 0.16, 0.16, 0.16],
        vertical_spacing=0.03,
        subplot_titles=("Fiyat / Trend", "Hacim", "RSI", "MACD")
    )
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Mum"), row=1, col=1)
    for col in ["SMA20","SMA50","SMA200","BB_UP","BB_LOW"]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Hacim"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"), row=3, col=1)
    fig.add_hline(y=70, row=3, col=1)
    fig.add_hline(y=30, row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], mode="lines", name="Signal"), row=4, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="Hist"), row=4, col=1)
    fig.update_layout(height=850, xaxis_rangeslider_visible=False, title=f"{symbol} PRO Teknik Grafik", template="plotly_white")
    return fig

def analyze(symbol, period, interval):
    df = load_data(symbol, period, interval)
    if df.empty or len(df) < 220:
        return None
    df = indicators(df)
    score, decision, notes = score_stock(df)
    support, resistance, stop, target1, target2, rr = target_stop(df)
    last = df.iloc[-1]
    return {
        "symbol": symbol, "df": df, "score": score, "decision": decision, "notes": notes,
        "support": support, "resistance": resistance, "stop": stop, "target1": target1, "target2": target2, "rr": rr,
        "price": float(last["Close"]), "rsi": float(last["RSI"]), "macd": float(last["MACD"]),
        "atr": float(last["ATR"]), "vol_ratio": float(last["VOL_RATIO"])
    }

st.sidebar.title("Ayarlar")
mode = st.sidebar.radio("Mod", ["Tek Hisse Analizi", "BIST Fırsat Tarayıcı"])
period = st.sidebar.selectbox("Veri dönemi", ["6mo","1y","2y","5y"], index=1)
interval = st.sidebar.selectbox("Periyot", ["1d","1wk"], index=0)

st.title("📈 BIST PRO Hisse Analiz Paneli")
st.caption("Teknik skor, alım bölgesi, hedef/stop, para girişi ve BIST fırsat tarayıcı.")
st.warning("Bu uygulama yatırım tavsiyesi vermez. Sinyaller teknik göstergelere dayalı otomatik yorumdur.")

if mode == "Tek Hisse Analizi":
    raw = st.sidebar.text_input("Hisse / Sembol", "THYAO")
    symbol = normalize_symbol(raw)
    res = analyze(symbol, period, interval)
    if res is None:
        st.error("Veri alınamadı veya veri yetersiz. Örn: THYAO, TUPRS, PEKGY")
        st.stop()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Son Fiyat", f"{res['price']:.2f}")
    c2.metric("PRO Skor", f"{res['score']}/100")
    c3.metric("RSI 14", f"{res['rsi']:.1f}")
    c4.metric("MACD", f"{res['macd']:.2f}")
    c5.metric("ATR 14", f"{res['atr']:.2f}")
    c6.metric("Hacim / Ort.", f"{res['vol_ratio']:.2f}x")

    st.subheader(res["decision"])

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Destek", f"{res['support']:.2f}")
    colB.metric("Direnç", f"{res['resistance']:.2f}")
    colC.metric("Hedef 1", f"{res['target1']:.2f}")
    colD.metric("Stop", f"{res['stop']:.2f}")

    st.plotly_chart(plot_chart(res["df"], symbol), use_container_width=True)

    st.subheader("Otomatik Teknik Yorum")
    for n in res["notes"]:
        st.write("• " + n)
    st.info(f"Risk/ödül oranı yaklaşık: {res['rr']:.2f}. Hedef 2: {res['target2']:.2f}")

else:
    st.subheader("🔎 BIST Fırsat Tarayıcı")
    custom = st.text_area("Taranacak hisseler", ",".join(BIST_WATCHLIST), height=120)
    symbols = [normalize_symbol(x.strip()) for x in custom.split(",") if x.strip()]
    if st.button("Taramayı Başlat"):
        rows = []
        prog = st.progress(0)
        for i, sym in enumerate(symbols):
            try:
                r = analyze(sym, period, interval)
                if r:
                    rows.append({
                        "Hisse": sym.replace(".IS",""),
                        "Karar": r["decision"],
                        "Skor": r["score"],
                        "Fiyat": round(r["price"],2),
                        "Hedef1": round(r["target1"],2),
                        "Stop": round(r["stop"],2),
                        "RSI": round(r["rsi"],1),
                        "Hacim/Ort": round(r["vol_ratio"],2),
                        "Risk/Ödül": round(r["rr"],2)
                    })
            except Exception:
                pass
            prog.progress((i+1)/len(symbols))
        out = pd.DataFrame(rows).sort_values(["Skor","Risk/Ödül"], ascending=False)
        st.dataframe(out, use_container_width=True)
        st.download_button("Sonuçları CSV indir", out.to_csv(index=False).encode("utf-8-sig"), "bist_firsat_tarayici.csv", "text/csv")

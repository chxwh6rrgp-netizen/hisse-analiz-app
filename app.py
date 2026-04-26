
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(page_title="BIST AI PRO Radar", layout="wide", page_icon="📈")

BIST_WATCHLIST = [
    "AEFES","AGHOL","AKBNK","AKCNS","AKFGY","AKSA","AKSEN","ALARK","ALBRK","ALGYO","ARCLK","ASELS","ASTOR",
    "BAGFS","BERA","BIMAS","BRSAN","BRYAT","BUCIM","CANTE","CCOLA","CIMSA","CLEBI","DOAS","DOHOL","ECILC",
    "EGEEN","EKGYO","ENJSA","ENKAI","EREGL","FROTO","GARAN","GENIL","GESAN","GLYHO","GUBRF","HALKB","HEKTS",
    "IPEKE","ISCTR","ISDMR","ISGYO","ISMEN","IZMDC","KARSN","KCHOL","KONTR","KORDS","KOZAA","KOZAL","KRDMD",
    "LOGO","MAVI","MGROS","ODAS","OYAKC","PETKM","PGSUS","QUAGR","SAHOL","SASA","SISE","SKBNK","SOKM","TAVHL",
    "TCELL","THYAO","TKFEN","TOASO","TSKB","TTKOM","TTRAK","TUPRS","TURSG","ULKER","VAKBN","VESTL","YKBNK",
    "ZOREN","ALFAS","CWENE","EUPWR","YEOTK","PEKGY","TRGYO","KONYA","NTHOL","PENTA","SELEC","TMSN","VAKKO",
    "AHGAZ","AKFYE","BIENY","BOBET","DAPGM","EUREN","GWIND","KCAER","KLSER","MIATK","REEDR","TABGD","TATEN",
    "ENERY","MAGEN","SMRTG","KMPUR","GESAN","SDTTR","HTTBT","KAYSE","FORTE","CWENE","IZENR","KZBGY","ULUUN"
]

def normalize_symbol(s):
    s = str(s).upper().strip().replace("İ","I")
    if not s.endswith(".IS"):
        s += ".IS"
    return s

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

@st.cache_data(ttl=900, show_spinner=False)
def load_single(symbol, period="2y", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df, n=14):
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    return line, signal, hist

def stochastic(df, n=14):
    low = df["Low"].rolling(n).min()
    high = df["High"].rolling(n).max()
    return 100 * (df["Close"] - low) / (high - low).replace(0, np.nan)

def compute_indicators(df):
    df = df.copy()
    df["SMA10"] = df["Close"].rolling(10).mean()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA100"] = df["Close"].rolling(100).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["RSI"] = rsi(df["Close"])
    df["ATR"] = atr(df)
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = macd(df["Close"])
    df["STOCH"] = stochastic(df)
    df["VOL_AVG20"] = df["Volume"].rolling(20).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_AVG20"].replace(0, np.nan)
    df["BB_MID"] = df["Close"].rolling(20).mean()
    df["BB_STD"] = df["Close"].rolling(20).std()
    df["BB_UP"] = df["BB_MID"] + 2 * df["BB_STD"]
    df["BB_LOW"] = df["BB_MID"] - 2 * df["BB_STD"]
    sign = np.sign(df["Close"].diff()).fillna(0)
    df["OBV"] = (sign * df["Volume"]).cumsum()
    df["RET5"] = df["Close"].pct_change(5) * 100
    df["RET20"] = df["Close"].pct_change(20) * 100
    df["RET60"] = df["Close"].pct_change(60) * 100
    df["HIGH20"] = df["High"].rolling(20).max()
    df["LOW20"] = df["Low"].rolling(20).min()
    df["HIGH60"] = df["High"].rolling(60).max()
    df["LOW60"] = df["Low"].rolling(60).min()
    return df.dropna()

def pct(a, b):
    if b == 0 or pd.isna(a) or pd.isna(b):
        return np.nan
    return (a / b - 1) * 100

def support_resistance(df, lookback=120):
    d = df.tail(lookback)
    swing_low = d["Low"].rolling(5).min().dropna()
    swing_high = d["High"].rolling(5).max().dropna()
    support = safe_float(swing_low.quantile(0.22), safe_float(d["Low"].min()))
    resistance = safe_float(swing_high.quantile(0.78), safe_float(d["High"].max()))
    major_low = safe_float(d["Low"].min())
    major_high = safe_float(d["High"].max())
    return support, resistance, major_low, major_high

def pattern_detection(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = last["Close"]
    patterns = []
    if price > last["HIGH20"] * 0.995 and last["VOL_RATIO"] > 1.2:
        patterns.append("20 günlük zirve kırılımı ve hacim teyidi")
    if last["RSI"] < 45 and last["MACD_HIST"] > prev["MACD_HIST"] and price > last["SMA10"]:
        patterns.append("dipten dönüş denemesi")
    if price > last["SMA20"] and prev["Close"] <= prev["SMA20"] and last["VOL_RATIO"] > 1.0:
        patterns.append("20 günlük ortalama üstüne hacimli dönüş")
    if last["BB_LOW"] * 0.98 <= price <= last["BB_LOW"] * 1.05 and last["RSI"] < 45:
        patterns.append("Bollinger alt bant tepki bölgesi")
    if price > last["SMA50"] and last["SMA20"] > last["SMA50"] and last["MACD"] > last["MACD_SIGNAL"]:
        patterns.append("trend devam formasyonu")
    return patterns

def professional_score(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = safe_float(last["Close"])
    atrv = max(safe_float(last["ATR"]), 0.01)
    support, resistance, major_low, major_high = support_resistance(df)
    dist_sup = pct(price, support)
    dist_res = pct(resistance, price)
    potential = max(dist_res, pct(price + 2.2 * atrv, price))
    stop = min(support * 0.985, price - 1.4 * atrv)
    target1 = max(resistance, price + 2.0 * atrv)
    target2 = price + 3.3 * atrv
    risk = max(price - stop, 0.01)
    reward = max(target1 - price, 0)
    rr = reward / risk if risk else np.nan

    # 5 ana modül: trend, momentum, para girişi, alım bölgesi, risk/ödül
    trend = 0
    if price > last["SMA20"] > last["SMA50"] > last["SMA100"]:
        trend = 24
        trend_label = "Ana trend güçlü; fiyat kısa ve orta vadeli ortalamaların üzerinde."
    elif price > last["SMA20"] > last["SMA50"]:
        trend = 20
        trend_label = "Kısa vadeli trend pozitif; 20 ve 50 günlük ortalamalar destek konumunda."
    elif price > last["SMA50"]:
        trend = 14
        trend_label = "Orta vadede toparlanma var fakat trend tam güçlenmiş değil."
    elif price > last["SMA20"]:
        trend = 9
        trend_label = "Kısa vadeli tepki var ama ana trend zayıf."
    else:
        trend = 3
        trend_label = "Fiyat önemli ortalamaların altında; trend zayıf."

    momentum = 0
    r = safe_float(last["RSI"])
    macd_pos = last["MACD"] > last["MACD_SIGNAL"]
    macd_turn = last["MACD_HIST"] > prev["MACD_HIST"]
    stoch = safe_float(last["STOCH"])
    if 50 <= r <= 65 and macd_pos and macd_turn:
        momentum = 24
        momentum_label = "Momentum sağlıklı; RSI aşırı alımda değil ve MACD güçleniyor."
    elif 45 <= r < 70 and macd_pos:
        momentum = 19
        momentum_label = "Momentum pozitif fakat güç teyidi için hacim/direnç takibi gerekir."
    elif 32 <= r < 45 and macd_turn and stoch < 55:
        momentum = 16
        momentum_label = "Dipten dönüş ihtimali var; erken sinyal oluşuyor."
    elif r >= 70:
        momentum = 6
        momentum_label = "RSI aşırı alım bölgesine yakın; yeni alımda kâr satışı riski yüksek."
    else:
        momentum = 5
        momentum_label = "Momentum net değil; alım için yeterli teyit yok."

    vol = safe_float(last["VOL_RATIO"], 0)
    obv20 = df["OBV"].iloc[-1] > df["OBV"].iloc[-20]
    obv60 = df["OBV"].iloc[-1] > df["OBV"].iloc[-60]
    money = 0
    if vol >= 1.7 and obv20:
        money = 22
        money_label = "Hacim ve OBV birlikte güçlü para girişine işaret ediyor."
    elif vol >= 1.2 and obv20:
        money = 17
        money_label = "Hacim destekli alıcı ilgisi var."
    elif obv60 and vol >= 0.9:
        money = 11
        money_label = "OBV uzun vadede toparlanıyor; para girişi ılımlı."
    else:
        money = 4
        money_label = "Hacim/OBV tarafında güçlü alıcı izi yok."

    zone = 0
    if dist_sup <= 4 and potential >= 7 and r < 68:
        zone = 18
        zone_label = "Fiyat desteğe yakın ve yukarı hedef alanı açık; alım bölgesi mantıklı."
    elif dist_sup <= 8 and potential >= 5:
        zone = 13
        zone_label = "Fiyat destekten çok uzak değil; kademeli takip edilebilir."
    elif dist_res <= 3:
        zone = 3
        zone_label = "Fiyat dirence yakın; yeni alım için güvenlik marjı dar."
    else:
        zone = 7
        zone_label = "Alım bölgesi net değil; daha iyi fiyat veya teyit beklenebilir."

    risk_score = 0
    if rr >= 1.6 and potential >= 8 and abs(pct(stop, price)) <= 8:
        risk_score = 18
        risk_label = "Risk/ödül cazip; stop mesafesi yönetilebilir."
    elif rr >= 1.15 and potential >= 5:
        risk_score = 12
        risk_label = "Risk/ödül izlenebilir ama kusursuz değil."
    elif abs(pct(stop, price)) > 10:
        risk_score = 4
        risk_label = "Stop mesafesi geniş; pozisyon riski yüksek."
    else:
        risk_score = 5
        risk_label = "Risk/ödül zayıf; hedefe göre risk fazla."

    total = int(min(100, trend + momentum + money + zone + risk_score))
    patterns = pattern_detection(df)

    # Profesyonel fırsat filtresi
    block_reasons = []
    if r >= 72: block_reasons.append("RSI aşırı alım bölgesinde")
    if dist_res <= 2.5: block_reasons.append("dirence çok yakın")
    if rr < 1.05: block_reasons.append("risk/ödül yetersiz")
    if potential < 5: block_reasons.append("hedef potansiyeli düşük")
    if vol < 0.75 and not obv20: block_reasons.append("hacim teyidi yok")

    real_opportunity = total >= 68 and len(block_reasons) == 0 and (macd_pos or macd_turn)

    if total >= 82 and real_opportunity:
        decision = "🔥 AI GÜÇLÜ FIRSAT"
        action = "Aday"
    elif total >= 68 and real_opportunity:
        decision = "🟢 AI ALIM BÖLGESİ"
        action = "Aday"
    elif total >= 55:
        decision = "🟡 AI TAKİP"
        action = "Bekle"
    else:
        decision = "🔴 AI RİSKLİ / ZAYIF"
        action = "Uzak Dur"

    return {
        "price": price, "score": total, "decision": decision, "action": action,
        "trend": trend, "momentum": momentum, "money": money, "zone": zone, "risk_score": risk_score,
        "trend_label": trend_label, "momentum_label": momentum_label, "money_label": money_label,
        "zone_label": zone_label, "risk_label": risk_label,
        "support": support, "resistance": resistance, "major_low": major_low, "major_high": major_high,
        "stop": stop, "target1": target1, "target2": target2, "rr": rr, "potential": pct(target1, price),
        "stop_pct": pct(stop, price), "dist_sup": dist_sup, "dist_res": dist_res,
        "rsi": r, "macd_pos": macd_pos, "macd_turn": macd_turn, "vol_ratio": vol,
        "patterns": patterns, "block_reasons": block_reasons, "real_opportunity": real_opportunity,
        "ret5": safe_float(last["RET5"],0), "ret20": safe_float(last["RET20"],0), "ret60": safe_float(last["RET60"],0),
    }

def ai_commentary(symbol, df, ctx):
    price = ctx["price"]
    lines = []

    if ctx["real_opportunity"]:
        opener = f"{symbol} şu an teknik olarak izlenebilir bir fırsat adayı. Bunun sebebi tek bir indikatör değil; trend, momentum, hacim ve risk/ödül tarafının aynı anda kabul edilebilir seviyede olması."
    elif ctx["action"] == "Bekle":
        opener = f"{symbol} için görüntü tamamen kötü değil fakat alım kararı için eksik teyitler var. Bu hisseyi acele almak yerine belirlenen seviyeler etrafında takip etmek daha mantıklı."
    else:
        opener = f"{symbol} şu an riskli bölgede. Teknik yapı yeterince desteklenmediği için alım tarafında acele etmek doğru görünmüyor."
    lines.append(opener)

    # Neden
    lines.append(f"Trend okuması: {ctx['trend_label']}")
    lines.append(f"Momentum okuması: {ctx['momentum_label']}")
    lines.append(f"Para girişi okuması: {ctx['money_label']}")
    lines.append(f"Alım bölgesi okuması: {ctx['zone_label']}")
    lines.append(f"Risk okuması: {ctx['risk_label']}")

    if ctx["patterns"]:
        lines.append("Yakalanan teknik yapı: " + ", ".join(ctx["patterns"]) + ".")
    else:
        lines.append("Belirgin bir kırılım/dip dönüş formasyonu henüz netleşmemiş.")

    # Senaryo
    bull = f"Olumlu senaryo: fiyat {ctx['resistance']:.2f} direncini hacimle geçerse ilk hedef {ctx['target1']:.2f}, devamında {ctx['target2']:.2f} bölgesi izlenebilir."
    bear = f"Olumsuz senaryo: fiyat {ctx['support']:.2f} desteğini kaybederse görünüm bozulur; risk azaltma seviyesi {ctx['stop']:.2f} civarıdır."
    lines.append(bull)
    lines.append(bear)

    if ctx["block_reasons"]:
        lines.append("Fırsat filtresini zayıflatan noktalar: " + ", ".join(ctx["block_reasons"]) + ".")

    # İnsan gibi karar cümlesi
    if "GÜÇLÜ" in ctx["decision"]:
        final = "Sonuç: Bu hisse 'hemen her fiyattan alınır' demek değildir; ama teknik sistem açısından güçlü adaylardan biridir. En mantıklı yaklaşım destek/stop planıyla kontrollü pozisyon olur."
    elif "ALIM" in ctx["decision"]:
        final = "Sonuç: Pozitif aday. Tam güçlenme için hacim ve direnç kırılımı takip edilmeli; stop seviyesi belirlenmeden işlem açılmamalı."
    elif "TAKİP" in ctx["decision"]:
        final = "Sonuç: İzleme listesinde kalmalı. Alım için ya desteğe yaklaşmasını ya da direnç üstü hacimli kapanış yapmasını beklemek daha sağlıklı."
    else:
        final = "Sonuç: Şu an para kazanma ihtimali kadar zarar riski de belirgin. Daha kaliteli fırsatlar varken öncelik verilmemeli."
    lines.append(final)
    return lines

def analyze(symbol, period="2y", interval="1d"):
    df = load_single(symbol, period, interval)
    if df.empty or len(df) < 230:
        return None
    df = compute_indicators(df)
    if df.empty or len(df) < 80:
        return None
    ctx = professional_score(df)
    comments = ai_commentary(symbol.replace(".IS",""), df, ctx)
    return {"symbol": symbol, "df": df, "ctx": ctx, "comments": comments}

def plot_chart(df, symbol, ctx):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.50, 0.16, 0.17, 0.17],
        vertical_spacing=0.035,
        subplot_titles=("Fiyat, Ortalamalar, Hedef/Stop", "Hacim", "RSI", "MACD")
    )
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Mum"), row=1, col=1)
    for col in ["SMA20","SMA50","SMA100","SMA200"]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_UP"], mode="lines", name="BB Üst", opacity=0.35), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOW"], mode="lines", name="BB Alt", opacity=0.35), row=1, col=1)
    fig.add_hline(y=ctx["support"], line_dash="dot", annotation_text="Destek", row=1, col=1)
    fig.add_hline(y=ctx["resistance"], line_dash="dot", annotation_text="Direnç", row=1, col=1)
    fig.add_hline(y=ctx["target1"], line_dash="dash", annotation_text="Hedef 1", row=1, col=1)
    fig.add_hline(y=ctx["stop"], line_dash="dash", annotation_text="Stop", row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Hacim"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"), row=3, col=1)
    fig.add_hline(y=70, row=3, col=1)
    fig.add_hline(y=50, row=3, col=1)
    fig.add_hline(y=30, row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], mode="lines", name="Signal"), row=4, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="Hist"), row=4, col=1)
    fig.update_layout(height=900, xaxis_rangeslider_visible=False, template="plotly_white", title=f"{symbol} AI PRO Grafik")
    return fig

def score_bar(ctx):
    data = pd.DataFrame({
        "Modül": ["Trend", "Momentum", "Para Girişi", "Alım Bölgesi", "Risk/Ödül"],
        "Puan": [ctx["trend"], ctx["momentum"], ctx["money"], ctx["zone"], ctx["risk_score"]]
    })
    fig = go.Figure(go.Bar(x=data["Modül"], y=data["Puan"]))
    fig.update_layout(height=300, template="plotly_white", yaxis_title="Puan")
    return fig

st.sidebar.title("📌 Panel")
mode = st.sidebar.radio("Mod", ["Tek Hisse AI Analiz", "BIST AI Radar", "Portföy / İzleme"])
period = st.sidebar.selectbox("Veri dönemi", ["1y","2y","5y"], index=1)
interval = st.sidebar.selectbox("Periyot", ["1d","1wk"], index=0)
st.sidebar.caption("Not: Veri Yahoo Finance üzerinden gelir. Bazı BIST hisselerinde veri gecikmeli/eksik olabilir.")

st.title("📈 BIST AI PRO Radar")
st.caption("Hızlı tarama + dashboard + insan gibi teknik yorum + fırsat/risk ayrımı.")
st.warning("Bu uygulama yatırım tavsiyesi değildir. Karar destek aracıdır. Stop, risk ve pozisyon büyüklüğü kullanıcı sorumluluğundadır.")

if mode == "Tek Hisse AI Analiz":
    raw = st.sidebar.text_input("Hisse", "THYAO")
    symbol = normalize_symbol(raw)
    with st.spinner(f"{symbol} analiz ediliyor..."):
        res = analyze(symbol, period, interval)
    if res is None:
        st.error("Veri alınamadı veya veri yetersiz. Sembolü kontrol et. Örn: THYAO, TUPRS, ASELS")
        st.stop()

    df, ctx = res["df"], res["ctx"]
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Fiyat", f"{ctx['price']:.2f}")
    m2.metric("AI Skor", f"{ctx['score']}/100")
    m3.metric("Karar", ctx["action"])
    m4.metric("Hedef 1", f"{ctx['target1']:.2f}", f"%{ctx['potential']:.1f}")
    m5.metric("Stop", f"{ctx['stop']:.2f}", f"%{ctx['stop_pct']:.1f}")
    m6.metric("Risk/Ödül", f"{ctx['rr']:.2f}")

    st.subheader(ctx["decision"])
    tab1, tab2, tab3 = st.tabs(["AI Yorum", "Grafik", "Skor Detayı"])

    with tab1:
        for line in res["comments"]:
            st.write(line)
        st.info(f"Destek: {ctx['support']:.2f} | Direnç: {ctx['resistance']:.2f} | RSI: {ctx['rsi']:.1f} | Hacim/Ort: {ctx['vol_ratio']:.2f}x")
    with tab2:
        st.plotly_chart(plot_chart(df, symbol, ctx), use_container_width=True)
    with tab3:
        st.plotly_chart(score_bar(ctx), use_container_width=True)
        st.write("Fırsat filtresini bloke eden sebepler:", ctx["block_reasons"] if ctx["block_reasons"] else "Yok")
        st.write("Yakalanan teknik yapılar:", ctx["patterns"] if ctx["patterns"] else "Net yapı yok")

elif mode == "BIST AI Radar":
    st.subheader("🚀 BIST AI Radar")
    st.caption("Geniş BIST listesini tarar. Önce gerçekten fırsat filtresinden geçenleri öne çıkarır.")
    preset = st.selectbox("Liste", ["Geniş BIST Listesi", "Kendi listem"])
    if preset == "Kendi listem":
        raw_list = st.text_area("Hisseleri virgülle yaz", "THYAO,TUPRS,ASELS,EREGL,KCHOL", height=120)
        symbols = [normalize_symbol(x) for x in raw_list.split(",") if x.strip()]
    else:
        symbols = [normalize_symbol(x) for x in BIST_WATCHLIST]

    colf1, colf2, colf3 = st.columns(3)
    min_score = colf1.slider("Minimum AI skor", 0, 100, 60)
    only_candidates = colf2.checkbox("Sadece adayları göster", value=True)
    min_potential = colf3.slider("Minimum hedef potansiyeli %", 0, 30, 5)

    if st.button("AI Radar Taramasını Başlat"):
        rows = []
        prog = st.progress(0)
        status = st.empty()
        for i, sym in enumerate(symbols):
            status.write(f"Taranıyor: {sym}")
            try:
                r = analyze(sym, period, interval)
                if r:
                    ctx = r["ctx"]
                    keep = ctx["score"] >= min_score and ctx["potential"] >= min_potential
                    if only_candidates:
                        keep = keep and ctx["real_opportunity"]
                    if keep:
                        rows.append({
                            "Hisse": sym.replace(".IS",""),
                            "Karar": ctx["decision"],
                            "Aksiyon": ctx["action"],
                            "Skor": ctx["score"],
                            "Fiyat": round(ctx["price"],2),
                            "Hedef1": round(ctx["target1"],2),
                            "Potansiyel %": round(ctx["potential"],1),
                            "Stop": round(ctx["stop"],2),
                            "Stop %": round(ctx["stop_pct"],1),
                            "Risk/Ödül": round(ctx["rr"],2),
                            "RSI": round(ctx["rsi"],1),
                            "Hacim/Ort": round(ctx["vol_ratio"],2),
                            "Trend": ctx["trend"],
                            "Momentum": ctx["momentum"],
                            "Para": ctx["money"],
                            "Kısa AI Yorum": r["comments"][0]
                        })
            except Exception:
                pass
            prog.progress((i+1)/len(symbols))
        status.write("Tarama tamamlandı.")
        if not rows:
            st.info("Bu filtrelere göre aday çıkmadı. Minimum skor/potansiyeli düşür veya 'sadece aday' filtresini kapat.")
        else:
            out = pd.DataFrame(rows).sort_values(["Skor","Risk/Ödül","Potansiyel %"], ascending=False)
            st.success(f"{len(out)} adet aday bulundu.")
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("En yüksek skor", int(out["Skor"].max()))
            k2.metric("Ortalama potansiyel", f"%{out['Potansiyel %'].mean():.1f}")
            k3.metric("En iyi R/R", f"{out['Risk/Ödül'].max():.2f}")
            k4.metric("Aday sayısı", len(out))
            st.dataframe(out, use_container_width=True, height=520)
            st.download_button("Radar sonucunu indir", out.to_csv(index=False).encode("utf-8-sig"), "bist_ai_radar.csv", "text/csv")

else:
    st.subheader("📒 Portföy / İzleme")
    st.caption("Kendi hisselerini yaz; uygulama hepsini tek tabloda yorumlasın.")
    raw_list = st.text_area("İzleme listen", "THYAO,TUPRS,ASELS,EREGL,KCHOL,PEKGY", height=120)
    symbols = [normalize_symbol(x) for x in raw_list.split(",") if x.strip()]
    if st.button("İzleme Listemi Analiz Et"):
        rows = []
        for sym in symbols:
            r = analyze(sym, period, interval)
            if r:
                ctx = r["ctx"]
                rows.append({
                    "Hisse": sym.replace(".IS",""),
                    "Karar": ctx["decision"],
                    "Skor": ctx["score"],
                    "Fiyat": round(ctx["price"],2),
                    "Hedef1": round(ctx["target1"],2),
                    "Stop": round(ctx["stop"],2),
                    "Risk/Ödül": round(ctx["rr"],2),
                    "AI Özet": r["comments"][-1]
                })
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("Skor", ascending=False), use_container_width=True)
        else:
            st.info("Veri alınamadı.")

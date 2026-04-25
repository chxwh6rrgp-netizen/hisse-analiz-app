
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="BIST PRO v2 Hisse Analiz Paneli", layout="wide")

# Geniş BIST tarama listesi. Yahoo Finance veri vermeyenleri uygulama otomatik pas geçer.
BIST_WATCHLIST = [
    "AEFES","AGHOL","AKBNK","AKCNS","AKFGY","AKSA","AKSEN","ALARK","ALBRK","ALGYO","ARCLK","ASELS","ASTOR",
    "BAGFS","BERA","BIMAS","BRSAN","BRYAT","BUCIM","CANTE","CCOLA","CIMSA","CLEBI","DOAS","DOHOL","ECILC",
    "EGEEN","EKGYO","ENJSA","ENKAI","EREGL","FROTO","GARAN","GENIL","GESAN","GLYHO","GUBRF","HALKB","HEKTS",
    "IPEKE","ISCTR","ISDMR","ISGYO","ISMEN","IZMDC","KARSN","KCHOL","KONTR","KORDS","KOZAA","KOZAL","KRDMD",
    "LOGO","MAVI","MGROS","ODAS","OYAKC","PETKM","PGSUS","QUAGR","SAHOL","SASA","SISE","SKBNK","SOKM","TAVHL",
    "TCELL","THYAO","TKFEN","TOASO","TSKB","TTKOM","TTRAK","TUPRS","TURSG","ULKER","VAKBN","VESTL","YKBNK",
    "ZOREN","ALFAS","CWENE","EUPWR","YEOTK","PEKGY","TRGYO","KONYA","NTHOL","PENTA","SELEC","TMSN","VAKKO"
]

def normalize_symbol(s):
    s = str(s).upper().strip().replace("İ","I")
    if not s.endswith(".IS"):
        s += ".IS"
    return s

@st.cache_data(ttl=900)
def load_data(symbol, period="2y", interval="1d"):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return df

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

def indicators(df):
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA100"] = df["Close"].rolling(100).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["RSI"] = rsi(df["Close"], 14)
    df["ATR"] = atr(df, 14)
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = macd(df["Close"])
    df["VOL_AVG20"] = df["Volume"].rolling(20).mean()
    df["VOL_RATIO"] = df["Volume"] / df["VOL_AVG20"].replace(0, np.nan)
    df["BB_MID"] = df["Close"].rolling(20).mean()
    df["BB_STD"] = df["Close"].rolling(20).std()
    df["BB_UP"] = df["BB_MID"] + 2 * df["BB_STD"]
    df["BB_LOW"] = df["BB_MID"] - 2 * df["BB_STD"]
    direction = np.sign(df["Close"].diff()).fillna(0)
    df["OBV"] = (direction * df["Volume"]).cumsum()
    df["RET20"] = df["Close"].pct_change(20) * 100
    df["RET60"] = df["Close"].pct_change(60) * 100
    return df.dropna()

def support_resistance(df, lookback=120):
    d = df.tail(lookback)
    support = float(d["Low"].rolling(5).min().dropna().quantile(0.22))
    resistance = float(d["High"].rolling(5).max().dropna().quantile(0.78))
    major_low = float(d["Low"].min())
    major_high = float(d["High"].max())
    return support, resistance, major_low, major_high

def pct(a, b):
    if b == 0 or pd.isna(b):
        return np.nan
    return (a / b - 1) * 100

def score_and_context(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(last["Close"])
    support, resistance, major_low, major_high = support_resistance(df)
    atrv = float(last["ATR"])
    dist_support = pct(price, support)
    dist_resistance = pct(resistance, price)
    trend_score = 0
    momentum_score = 0
    volume_score = 0
    risk_score = 0

    # Trend değerlendirmesi
    if price > last["SMA20"] > last["SMA50"] > last["SMA100"]:
        trend_score = 28
        trend_state = "güçlü yükselen trend"
    elif price > last["SMA20"] > last["SMA50"]:
        trend_score = 22
        trend_state = "pozitif kısa vadeli trend"
    elif price > last["SMA50"]:
        trend_score = 15
        trend_state = "orta vadede toparlanma eğilimi"
    elif price > last["SMA20"]:
        trend_score = 10
        trend_state = "kısa vadeli tepki denemesi"
    else:
        trend_score = 4
        trend_state = "zayıf trend"

    # Momentum
    r = float(last["RSI"])
    macd_positive = last["MACD"] > last["MACD_SIGNAL"]
    macd_improving = last["MACD_HIST"] > prev["MACD_HIST"]
    if 50 <= r <= 65 and macd_positive and macd_improving:
        momentum_score = 28
        momentum_state = "sağlıklı ve güçlenen momentum"
    elif 45 <= r < 70 and macd_positive:
        momentum_score = 22
        momentum_state = "pozitif momentum"
    elif 35 <= r < 45 and macd_improving:
        momentum_score = 16
        momentum_state = "dipten toparlanma denemesi"
    elif r >= 70:
        momentum_score = 8
        momentum_state = "aşırı alıma yaklaşmış momentum"
    else:
        momentum_score = 6
        momentum_state = "zayıf veya kararsız momentum"

    # Hacim / para girişi
    vol_ratio = float(last["VOL_RATIO"]) if not pd.isna(last["VOL_RATIO"]) else 0
    obv_up_20 = df["OBV"].iloc[-1] > df["OBV"].iloc[-20]
    obv_up_60 = df["OBV"].iloc[-1] > df["OBV"].iloc[-60]
    if vol_ratio >= 1.6 and obv_up_20:
        volume_score = 22
        volume_state = "hacim destekli para girişi işareti"
    elif vol_ratio >= 1.1 and obv_up_20:
        volume_score = 16
        volume_state = "ılımlı hacim desteği"
    elif obv_up_60:
        volume_score = 10
        volume_state = "uzun vadeli OBV toparlanması"
    else:
        volume_score = 4
        volume_state = "hacim desteği zayıf"

    # Risk/ödül
    stop = min(support * 0.985, price - 1.35 * atrv)
    target1 = max(resistance, price + 1.8 * atrv)
    target2 = price + 3.0 * atrv
    downside = max(price - stop, 0.01)
    upside = target1 - price
    rr = upside / downside if downside > 0 else np.nan
    potential = pct(target1, price)
    stop_loss_pct = pct(stop, price)

    if potential >= 8 and rr >= 1.4 and dist_support <= 8:
        risk_score = 22
        risk_state = "risk/ödül dengesi olumlu"
    elif potential >= 5 and rr >= 1.1:
        risk_score = 15
        risk_state = "risk/ödül dengesi izlenebilir"
    elif dist_resistance < 3:
        risk_score = 5
        risk_state = "dirence çok yakın, marj dar"
    else:
        risk_score = 8
        risk_state = "risk/ödül net güçlü değil"

    total = int(min(100, trend_score + momentum_score + volume_score + risk_score))

    # Fırsat filtresi: her yüksek skora fırsat denmez
    real_opportunity = (
        total >= 65 and
        potential >= 5 and
        rr >= 1.05 and
        r < 72 and
        (macd_positive or macd_improving) and
        dist_resistance > 2
    )

    if total >= 80 and real_opportunity:
        decision = "🔥 GÜÇLÜ FIRSAT"
    elif total >= 65 and real_opportunity:
        decision = "🟢 ALIM BÖLGESİ / POZİTİF"
    elif total >= 50:
        decision = "🟡 TAKİP"
    else:
        decision = "🔴 ZAYIF / UZAK DUR"

    return {
        "score": total, "decision": decision,
        "trend_state": trend_state, "momentum_state": momentum_state, "volume_state": volume_state, "risk_state": risk_state,
        "support": support, "resistance": resistance, "major_low": major_low, "major_high": major_high,
        "stop": stop, "target1": target1, "target2": target2, "rr": rr, "potential": potential, "stop_loss_pct": stop_loss_pct,
        "dist_support": dist_support, "dist_resistance": dist_resistance,
        "real_opportunity": real_opportunity
    }

def human_comment(symbol, df, ctx):
    last = df.iloc[-1]
    price = float(last["Close"])
    r = float(last["RSI"])
    vol = float(last["VOL_RATIO"]) if not pd.isna(last["VOL_RATIO"]) else 0
    macd_positive = last["MACD"] > last["MACD_SIGNAL"]
    macd_improving = last["MACD_HIST"] > df["MACD_HIST"].iloc[-2]
    ret20 = float(last["RET20"]) if not pd.isna(last["RET20"]) else 0
    ret60 = float(last["RET60"]) if not pd.isna(last["RET60"]) else 0

    # Trend yorumu
    if price > last["SMA20"] > last["SMA50"]:
        trend_text = f"{symbol} tarafında fiyat kısa vadeli ortalamaların üzerinde kaldığı için ana görüntü pozitif. Son 20 günlük performans yaklaşık %{ret20:.1f}; bu, hissenin son dönemde piyasadan ilgi gördüğünü gösterir."
    elif price > last["SMA50"]:
        trend_text = f"{symbol} 50 günlük ortalamanın üzerinde tutunuyor fakat kısa vadeli yapı tam anlamıyla güçlü değil. Bu görünüm genelde 'toparlanma var ama teyit gerekiyor' şeklinde okunur."
    elif price < last["SMA20"] and price < last["SMA50"]:
        trend_text = f"{symbol} şu anda hem kısa hem orta vadeli ortalamaların altında. Bu nedenle trend tarafında acele alım yerine dönüş teyidi beklemek daha sağlıklı olur."
    else:
        trend_text = f"{symbol} tarafında trend karışık. Fiyat bazı ortalamaların üzerinde olsa da net bir yukarı trend görüntüsü henüz oluşmamış."

    # Momentum yorumu
    if 50 <= r <= 65 and macd_positive:
        momentum_text = f"RSI {r:.1f} seviyesinde; bu bölge aşırı alım olmadan pozitif momentum anlamına gelir. MACD de pozitif olduğu için teknik görünüm destekleniyor."
    elif r > 70:
        momentum_text = f"RSI {r:.1f} ile aşırı alım bölgesine yaklaşmış. Hisse güçlü olabilir fakat bu seviyelerde kısa vadeli kâr satışı riski artar."
    elif 35 <= r < 45 and macd_improving:
        momentum_text = f"RSI {r:.1f} ile düşük bölgede ve MACD histogramı toparlanıyor. Bu yapı dipten dönüş denemesi olarak izlenebilir, ancak teyit için hacim önemlidir."
    else:
        momentum_text = f"RSI {r:.1f} ve MACD birlikte çok net bir güç sinyali üretmiyor. Momentum tarafında seçici olmak gerekir."

    # Hacim yorumu
    if vol >= 1.6:
        volume_text = f"Hacim son 20 günlük ortalamanın {vol:.2f} katı. Bu durum hareketin sıradan olmadığını, piyasada belirgin ilgi oluştuğunu gösterir."
    elif vol >= 1.0:
        volume_text = f"Hacim ortalamanın biraz üzerinde ({vol:.2f}x). Hareket destekleniyor ama çok güçlü para girişi demek için daha yüksek hacim görmek iyi olur."
    else:
        volume_text = f"Hacim ortalamanın altında ({vol:.2f}x). Bu nedenle yükseliş varsa bile hacimle teyit edilmediği için temkinli okunmalı."

    # Destek/direnç ve hedef yorumu
    if ctx["dist_support"] <= 4 and ctx["potential"] >= 6:
        zone_text = f"Fiyat desteğe yakın ve hedef potansiyeli %{ctx['potential']:.1f}. Bu yüzden teknik olarak alım bölgesine yakın sayılabilir."
    elif ctx["dist_resistance"] <= 3:
        zone_text = f"Fiyat dirence çok yakın. Hedef alanı daraldığı için yeni alımda risk/ödül cazibesi zayıflıyor."
    elif ctx["potential"] >= 8 and ctx["rr"] >= 1.3:
        zone_text = f"Hedef alanı geniş: ilk hedefe potansiyel yaklaşık %{ctx['potential']:.1f}, risk/ödül oranı {ctx['rr']:.2f}. Bu teknik açıdan izlenebilir bir fırsat yapısıdır."
    else:
        zone_text = f"Destek-direnç mesafesi çok ideal değil. İlk hedef potansiyeli %{ctx['potential']:.1f}, risk/ödül oranı {ctx['rr']:.2f}; bu nedenle aceleci olmamak gerekir."

    # Sonuç
    if "GÜÇLÜ FIRSAT" in ctx["decision"]:
        final_text = "Genel sonuç: hisse teknik olarak güçlü fırsat bölgesinde görünüyor; yine de stop seviyesi mutlaka izlenmeli."
    elif "ALIM BÖLGESİ" in ctx["decision"]:
        final_text = "Genel sonuç: teknik görünüm pozitif. Alım için makul bölge oluşmuş olabilir fakat hacim ve destek korunumu takip edilmeli."
    elif "TAKİP" in ctx["decision"]:
        final_text = "Genel sonuç: hisse izlemeye değer, fakat şu an net alım sinyali yeterince güçlü değil."
    else:
        final_text = "Genel sonuç: teknik yapı zayıf. Daha net dönüş veya hacim teyidi gelmeden riskli görünüyor."

    return [trend_text, momentum_text, volume_text, zone_text, final_text]

def plot_chart(df, symbol, ctx):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.52, 0.16, 0.16, 0.16],
        vertical_spacing=0.03,
        subplot_titles=("Fiyat / Trend", "Hacim", "RSI", "MACD")
    )
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Mum"), row=1, col=1)
    for col in ["SMA20","SMA50","SMA100","SMA200","BB_UP","BB_LOW"]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col), row=1, col=1)
    fig.add_hline(y=ctx["support"], line_dash="dot", annotation_text="Destek", row=1, col=1)
    fig.add_hline(y=ctx["resistance"], line_dash="dot", annotation_text="Direnç", row=1, col=1)
    fig.add_hline(y=ctx["target1"], line_dash="dash", annotation_text="Hedef 1", row=1, col=1)
    fig.add_hline(y=ctx["stop"], line_dash="dash", annotation_text="Stop", row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Hacim"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"), row=3, col=1)
    fig.add_hline(y=70, row=3, col=1)
    fig.add_hline(y=30, row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], mode="lines", name="Signal"), row=4, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="Hist"), row=4, col=1)
    fig.update_layout(height=880, xaxis_rangeslider_visible=False, title=f"{symbol} PRO v2 Teknik Grafik", template="plotly_white")
    return fig

def analyze(symbol, period, interval):
    df = load_data(symbol, period, interval)
    if df.empty or len(df) < 220:
        return None
    df = indicators(df)
    if df.empty or len(df) < 30:
        return None
    ctx = score_and_context(df)
    last = df.iloc[-1]
    comments = human_comment(symbol.replace(".IS",""), df, ctx)
    return {
        "symbol": symbol, "df": df, "ctx": ctx, "comments": comments,
        "price": float(last["Close"]), "rsi": float(last["RSI"]), "macd": float(last["MACD"]),
        "atr": float(last["ATR"]), "vol_ratio": float(last["VOL_RATIO"]) if not pd.isna(last["VOL_RATIO"]) else 0
    }

st.sidebar.title("Ayarlar")
mode = st.sidebar.radio("Mod", ["Tek Hisse Analizi", "BIST Fırsat Tarayıcı"])
period = st.sidebar.selectbox("Veri dönemi", ["1y","2y","5y"], index=1)
interval = st.sidebar.selectbox("Periyot", ["1d","1wk"], index=0)

st.title("📈 BIST PRO v2 Hisse Analiz Paneli")
st.caption("Daha anlaşılır yorum, gerçek fırsat filtresi, geniş BIST tarama, hedef/stop ve risk/ödül analizi.")
st.warning("Bu uygulama yatırım tavsiyesi vermez. Sonuçlar teknik göstergelere dayalı otomatik analizdir.")

if mode == "Tek Hisse Analizi":
    raw = st.sidebar.text_input("Hisse / Sembol", "THYAO")
    symbol = normalize_symbol(raw)
    res = analyze(symbol, period, interval)
    if res is None:
        st.error("Veri alınamadı veya veri yetersiz. Örn: THYAO, TUPRS, PEKGY")
        st.stop()

    ctx = res["ctx"]
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Son Fiyat", f"{res['price']:.2f}")
    c2.metric("PRO Skor", f"{ctx['score']}/100")
    c3.metric("RSI 14", f"{res['rsi']:.1f}")
    c4.metric("MACD", f"{res['macd']:.2f}")
    c5.metric("ATR 14", f"{res['atr']:.2f}")
    c6.metric("Hacim / Ort.", f"{res['vol_ratio']:.2f}x")

    st.subheader(ctx["decision"])

    colA, colB, colC, colD, colE = st.columns(5)
    colA.metric("Destek", f"{ctx['support']:.2f}")
    colB.metric("Direnç", f"{ctx['resistance']:.2f}")
    colC.metric("Hedef 1", f"{ctx['target1']:.2f}", f"%{ctx['potential']:.1f}")
    colD.metric("Stop", f"{ctx['stop']:.2f}", f"%{ctx['stop_loss_pct']:.1f}")
    colE.metric("Risk/Ödül", f"{ctx['rr']:.2f}")

    st.plotly_chart(plot_chart(res["df"], symbol, ctx), use_container_width=True)

    st.subheader("Düşünülmüş Teknik Yorum")
    for p in res["comments"]:
        st.write(p)

    with st.expander("Skorun nasıl oluştuğunu göster"):
        st.write(f"Trend: {ctx['trend_state']}")
        st.write(f"Momentum: {ctx['momentum_state']}")
        st.write(f"Hacim / OBV: {ctx['volume_state']}")
        st.write(f"Risk/Ödül: {ctx['risk_state']}")
        st.write(f"Desteğe uzaklık: %{ctx['dist_support']:.1f}")
        st.write(f"Dirence uzaklık: %{ctx['dist_resistance']:.1f}")

else:
    st.subheader("🔎 Geniş BIST Fırsat Tarayıcı")
    st.caption("Tüm listeyi tarar; veri gelmeyen hisseleri pas geçer. Sonuçlar fırsat filtresine göre sıralanır.")
    custom = st.text_area("Taranacak hisseler", ",".join(BIST_WATCHLIST), height=160)
    symbols = [normalize_symbol(x.strip()) for x in custom.split(",") if x.strip()]
    only_opps = st.checkbox("Sadece gerçek fırsat filtresinden geçenleri göster", value=True)

    if st.button("Tüm Listeyi Tara"):
        rows = []
        prog = st.progress(0)
        status = st.empty()
        for i, sym in enumerate(symbols):
            status.write(f"Taranıyor: {sym}")
            try:
                r = analyze(sym, period, interval)
                if r:
                    ctx = r["ctx"]
                    if (not only_opps) or ctx["real_opportunity"]:
                        rows.append({
                            "Hisse": sym.replace(".IS",""),
                            "Karar": ctx["decision"],
                            "Skor": ctx["score"],
                            "Fiyat": round(r["price"],2),
                            "Hedef1": round(ctx["target1"],2),
                            "Potansiyel %": round(ctx["potential"],1),
                            "Stop": round(ctx["stop"],2),
                            "Stop %": round(ctx["stop_loss_pct"],1),
                            "RSI": round(r["rsi"],1),
                            "Hacim/Ort": round(r["vol_ratio"],2),
                            "Risk/Ödül": round(ctx["rr"],2),
                            "Kısa Yorum": r["comments"][-1]
                        })
            except Exception:
                pass
            prog.progress((i+1)/len(symbols))
        status.write("Tarama tamamlandı.")
        if rows:
            out = pd.DataFrame(rows).sort_values(["Skor","Risk/Ödül","Potansiyel %"], ascending=False)
            st.dataframe(out, use_container_width=True)
            st.download_button("Sonuçları CSV indir", out.to_csv(index=False).encode("utf-8-sig"), "bist_pro_v2_firsatlar.csv", "text/csv")
        else:
            st.info("Bu filtrelere göre fırsat bulunamadı. 'Sadece gerçek fırsat' filtresini kaldırıp tekrar deneyebilirsin.")

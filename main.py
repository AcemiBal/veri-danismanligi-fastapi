import os
import io
import json
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import requests

from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Float,
    Boolean,
    ForeignKey,
    Text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# --- Yol & klasörler ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
CHART_DIR = os.path.join(STATIC_DIR, "charts")
REPORT_DIR = os.path.join(STATIC_DIR, "reports")

# PDF için Türkçe karakter desteği olan font kaydı
# Windows ortamında Arial kullanılmaya çalışılır, bulunamazsa Helvetica'ya düşer.
try:
    pdfmetrics.registerFont(TTFont("ArialTR", "C:/Windows/Fonts/arial.ttf"))
    PDF_FONT = "ArialTR"
except Exception:
    PDF_FONT = "Helvetica"

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# --- Veritabanı ---

DATABASE_URL = "sqlite:///" + os.path.join(BASE_DIR, "veridanismanligi.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    # Temel giriş bilgileri
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)  # Demo için düz şifre, prod için hash önerilir
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Profil / iletişim bilgileri
    full_name = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    company = Column(String, nullable=True)
    sector = Column(String, nullable=True)

    uploads = relationship("Upload", back_populates="user")



class Upload(Base):
    __tablename__ = "uploads"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    file_name = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    company = Column(String, nullable=True)

    # İletişim bilgileri (login yok, sadece upload anında alınır)
    contact_name = Column(String, nullable=True)
    contact_phone = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    contact_sector = Column(String, nullable=True)

    row_count = Column(Integer, default=0)
    col_count = Column(Integer, default=0)
    total_cells = Column(Integer, default=0)
    total_missing = Column(Integer, default=0)
    quality_score = Column(Float, default=0.0)
    top_missing_col = Column(String, nullable=True)
    top_var_col = Column(String, nullable=True)
    domain_insights = Column(Text, nullable=True)

    ai_summary = Column(Text, nullable=True)
    ai_risks = Column(Text, nullable=True)
    ai_features = Column(Text, nullable=True)
    ai_models = Column(Text, nullable=True)
    ai_recommendations = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="uploads")


Base.metadata.create_all(bind=engine)

# --- FastAPI app ---

app = FastAPI(title="Veri Danışmanlığı – Akıllı Veri Analiz Paneli")
app.add_middleware(SessionMiddleware, secret_key="CHANGE_THIS_SECRET")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- Bellek içi cache (son analizler) ---

ANALYSIS_CACHE: Dict[int, Dict[str, Any]] = {}


# --- DB dependency & yardımcılar ---

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def current_user(request: Request, db: Session) -> Optional[User]:
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return db.query(User).filter(User.id == user_id).first()


# --- AI analizi (OpenAI, internetli) ---

def ai_analyze_dataframe(df: pd.DataFrame) -> Dict[str, str]:
    """
    OpenAI (internetli) GPT-4o-mini ile veri seti analizi.
    Çıkış: Türkçe metinler içeren JSON alanları.
    """
    OPENAI_API_KEY = "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"  # <- sen dolduracaksın

    rows, cols = df.shape
    missing_total = int(df.isna().sum().sum())
    top_missing_col = df.isna().sum().idxmax() if cols > 0 else ""

    prompt = f"""
Sen kıdemli bir veri bilimcisin. Aşağıdaki veri setini TÜRKÇE olarak değerlendir:

- Satır sayısı: {rows}
- Kolon sayısı: {cols}
- Toplam eksik hücre: {missing_total}
- En çok eksik veri olan kolon: {top_missing_col}

Çıktıyı SADECE geçerli bir JSON NESNESİ olarak üret.
Tüm açıklamalar profesyonel ve akıcı TÜRKÇE olsun.

Şu şemaya TAM uy:

{{
  "summary": "Veri setinin genel Türkçe özeti (2–4 cümle).",
  "risks": "Veri kalitesi ve modelleme ile ilgili başlıca riskleri Türkçe açıklayan paragraf veya madde listesi.",
  "features": "Özellik mühendisliği (feature engineering) için Türkçe öneriler.",
  "ml_models": "Bu veri seti için uygun makine öğrenmesi modellerini Türkçe olarak açıklayan metin.",
  "recommendations": "Şirket / iş birimi için Türkçe aksiyon ve sonraki adım önerileri."
}}

ÖNEMLİ KURALLAR:
- JSON dışında hiçbir açıklama YAZMA.
- markdown kullanma, ```json veya ``` yazma.
- Anahtar adları İNGİLİZCE kalacak (summary, risks, features, ml_models, recommendations),
  fakat TÜM METİN DEĞERLERİ TÜRKÇE olacak.
""".strip()

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }

    # API anahtarı yoksa demo mod
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("BURAYA_"):
        return {
            "summary": "Demo mod: OpenAI anahtarı tanımlanmadığı için basitleştirilmiş yerel özet gösteriliyor.",
            "risks": "Gerçek zamanlı AI analizi devre dışı. Eksik veri, aykırı değerler ve iş senaryosu özel riskler için manuel kontrol önerilir.",
            "features": "Tarih, kategori ve sayısal alanlar üzerinden türetilmiş özellikler (oranlar, segmentler, zaman kırılımları) tasarlanabilir.",
            "ml_models": "Regresyon, sınıflandırma ve kümelendirme algoritmaları veri yapısına göre değerlendirilebilir.",
            "recommendations": "Gerçek AI analizi için OpenAI API anahtarı tanımlanmalı ve sistem internet erişimi ile yeniden çalıştırılmalıdır.",
        }

    try:
        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=60,
            proxies={"http": None, "https": None},  # kurumsal proxy bypass
        )

        try:
            data = resp.json()
        except Exception:
            return {
                "summary": "AI çalıştırılamadı.",
                "risks": f"OpenAI'den geçersiz/boş cevap geldi: {resp.text[:200]}",
                "features": "-",
                "ml_models": "-",
                "recommendations": "-",
            }

        if "error" in data:
            return {
                "summary": "AI çalıştırılamadı.",
                "risks": data["error"].get("message", "OpenAI API hatası"),
                "features": "-",
                "ml_models": "-",
                "recommendations": "-",
            }

        ai_text = data["choices"][0]["message"]["content"].strip()

        # ```json bloklarını temizle
        if ai_text.startswith("```"):
            first_nl = ai_text.find("\n")
            if first_nl != -1:
                ai_text = ai_text[first_nl + 1 :].strip()
            if ai_text.endswith("```"):
                ai_text = ai_text[:-3].strip()

        try:
            return json.loads(ai_text)
        except Exception as e:
            return {
                "summary": "AI JSON analiz hatası.",
                "risks": f"Cevap JSON formatında değil: {e}\nCevap: {ai_text[:300]}",
                "features": "-",
                "ml_models": "-",
                "recommendations": "-",
            }

    except Exception as e:
        return {
            "summary": "AI analizi çalışmadı.",
            "risks": f"Ağ / proxy hatası: {e}",
            "features": "-",
            "ml_models": "-",
            "recommendations": "-",
        }


# --- Grafik üretimi ---

def generate_charts(df: pd.DataFrame, upload_id: int) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    hist_paths: List[str] = []
    trend_url: Optional[str] = None

    # Histogramlar
    for col in numeric_cols:
        plt.figure()
        df[col].dropna().hist(bins=30)
        plt.title(f"{col} - Dağılım")
        plt.xlabel(col)
        plt.ylabel("Frekans")
        plt.tight_layout()

        filename = f"{upload_id}_hist_{col}.png"
        filepath = os.path.join(CHART_DIR, filename)
        plt.savefig(filepath)
        plt.close()

        hist_paths.append(f"/static/charts/{filename}")

    # Basit trend (ilk sayısal kolona göre)
    if numeric_cols:
        col = numeric_cols[0]
        plt.figure()
        df[col].reset_index(drop=True).plot()
        plt.title(f"{col} - Trend")
        plt.xlabel("Index")
        plt.ylabel(col)
        plt.tight_layout()

        filename = f"{upload_id}_trend_{col}.png"
        filepath = os.path.join(CHART_DIR, filename)
        plt.savefig(filepath)
        plt.close()

        trend_url = f"/static/charts/{filename}"

    return {"histograms": hist_paths, "trend": trend_url}


def build_chart_cards(charts: Dict[str, Any]) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    if not charts:
        return cards

    histos = charts.get("histograms") or []
    for url in histos:
        base = os.path.basename(url)
        name = os.path.splitext(base)[0]
        parts = name.split("_")
        col = parts[-1] if len(parts) > 2 else ""
        title = f"{col} – Dağılım" if col else "Dağılım Grafiği"
        cards.append({"title": title, "url": url})

    trend_url = charts.get("trend")
    if trend_url:
        base = os.path.basename(trend_url)
        name = os.path.splitext(base)[0]
        parts = name.split("_")
        col = parts[-1] if len(parts) > 2 else ""
        title = f"{col} – Trend" if col else "Trend Grafiği"
        cards.append({"title": title, "url": trend_url})

    return cards


# --- PDF üretimi (AI odaklı rapor) ---

def generate_pdf_report(
    output_path: str,
    summary: str,
    risks: str,
    features: str,
    models: str,
    recommendations: str,
    chart_files: Optional[List[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Daha düzenli, UX/UI odaklı PDF rapor üretir.

    Sayfa 1:
      - Başlık, tarih
      - Firma / müşteri kutusu (meta ile)
      - AI Özet, Riskler, Feature Engineering, Modeller, Aksiyonlar

    Sonraki sayfalar:
      - 'Grafikler' başlığı
      - Her sayfada birden fazla grafik, tutarlı başlıklarla
    """
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    import os
    import textwrap
    from datetime import datetime

    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    margin = 2 * cm

    # Küçük yardımcı fonksiyonlar
    def new_page_header(title: str) -> float:
        """Yeni sayfa açar ve başlık basar, kullanılacak y koordinatını döner."""
        c.showPage()
        c.setFont(PDF_FONT, 16)
        y_ = height - margin
        c.drawString(margin, y_, title)
        return y_ - 1.2 * cm

    def draw_section_title(txt: str, y_: float) -> float:
        if y_ < 3 * cm:
            y_ = new_page_header("Veri Analiz Raporu")
        c.setFont(PDF_FONT, 12)
        c.drawString(margin, y_, txt)
        c.setLineWidth(0.3)
        c.line(margin, y_ - 0.15 * cm, width - margin, y_ - 0.15 * cm)
        return y_ - 0.6 * cm

    def draw_paragraph(text: str, y_: float, font_size: int = 10) -> float:
        """Basit paragraf çizimi, otomatik satır kırma ve sayfa devamı."""
        if not text:
            return y_
        c.setFont(PDF_FONT, font_size)
        max_chars = 110  # yaklaşık, sayfa genişliğine göre
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                y_ -= 0.4 * cm
                continue
            wrapped = textwrap.wrap(line, max_chars) or [line]
            for wline in wrapped:
                if y_ < 2.5 * cm:
                    # yeni sayfa
                    y_ = new_page_header("Veri Analiz Raporu (devam)")
                c.drawString(margin, y_, wline)
                y_ -= 0.45 * cm
        y_ -= 0.3 * cm
        return y_

    # ---------- SAYFA 1: Başlık + Tarih ----------
    c.setFont(PDF_FONT, 18)
    y = height - margin
    c.drawString(margin, y, "Veri Analiz Raporu")

    c.setFont(PDF_FONT, 9)
    created_text = f"Oluşturulma Tarihi: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} (UTC)"
    c.drawRightString(width - margin, y, created_text)

    y -= 1.2 * cm

    # ---------- Müşteri / Firma Bilgileri Kutusu ----------
    if meta:
        c.setFont(PDF_FONT, 11)
        box_top = y
        box_bottom = y - 3.2 * cm
        if box_bottom < 2 * cm:
            box_bottom = 2 * cm
        # Çerçeve
        c.setLineWidth(0.6)
        c.rect(margin, box_bottom, width - 2 * margin, box_top - box_bottom, stroke=1, fill=0)

        y_line = box_top - 0.8 * cm

        def meta_line(label: str, key: str):
            nonlocal y_line
            val = (meta.get(key) or "").strip()
            if not val:
                return
            c.drawString(margin + 0.4 * cm, y_line, f"{label}: {val}")
            y_line -= 0.55 * cm

        meta_line("Firma", "company")
        meta_line("Ad Soyad", "contact_name")
        meta_line("E-posta", "contact_email")
        meta_line("Telefon", "contact_phone")
        meta_line("Sektör", "contact_sector")

        y = box_bottom - 0.8 * cm
    else:
        y -= 0.4 * cm

    # ---------- AI Bölümleri ----------
    sections = [
        ("AI Özet", summary),
        ("Riskler", risks),
        ("Feature Engineering Önerileri", features),
        ("Uygun ML Modelleri", models),
        ("Aksiyon Önerileri", recommendations),
    ]

    for title, text in sections:
        if text and text.strip():
            y = draw_section_title(title, y)
            y = draw_paragraph(text, y)

    # ---------- GRAFİKLER ----------
    if chart_files:
        # Yeni sayfa başlığı
        y = new_page_header("Grafikler")
        c.setFont(PDF_FONT, 10)

        for idx, chart_path in enumerate(chart_files, start=1):
            if y < 8 * cm:
                y = new_page_header("Grafikler")

            chart_name = os.path.basename(chart_path)
            c.setFont(PDF_FONT, 11)
            c.drawString(margin, y, f"Grafik {idx}: {chart_name}")
            y -= 0.6 * cm

            try:
                # Grafik alanı: sayfayı ortalayarak, orantıyı koruyarak
                img_height = 9 * cm
                img_width = width - 2 * margin
                c.drawImage(
                    chart_path,
                    margin,
                    y - img_height,
                    width=img_width,
                    height=img_height,
                    preserveAspectRatio=True,
                    anchor="n",
                )
                y -= img_height + 1 * cm
            except Exception:
                c.setFont(PDF_FONT, 9)
                c.drawString(margin, y, "(Grafik dosyası okunamadı)")
                y -= 1 * cm

    c.save()


# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
def index(request: Request, db: Session = Depends(get_db)):
    user = current_user(request, db)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "user": user},
    )


@app.get("/register", response_class=HTMLResponse)
def register_get(request: Request):
    return templates.TemplateResponse("register.html", {"request": request, "error": None})


@app.post("/register", response_class=HTMLResponse)
def register_post(
    request: Request,
    full_name: str = Form(...),
    phone: str = Form(...),
    company_name: str = Form(""),
    sector: str = Form(""),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    """Kullanıcı kaydı: kullanıcı rolü (is_admin=False) ve profil bilgileri."""
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Bu e-posta ile zaten bir kullanıcı var."},
        )

    user = User(
        email=email,
        password=password,
        is_admin=False,  # Arayüzden admin kaydı alınmıyor
        full_name=full_name,
        phone=phone,
        company=company_name,
        sector=sector,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    request.session["user_id"] = user.id
    redirect_url = "/dashboard"
    return RedirectResponse(url=redirect_url, status_code=302)


@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request):
    """Standart kullanıcı girişi (admin olmayan kullanıcılar için)."""
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login", response_class=HTMLResponse)
def login_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = (
        db.query(User)
        .filter(User.email == email, User.password == password)
        .first()
    )
    if not user:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Geçersiz e-posta veya şifre."},
        )

    # Admin buradan girerse sadece kullanıcı dashboard'una yönlendirilir
    request.session["user_id"] = user.id
    return RedirectResponse(url="/dashboard", status_code=302)


@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_get(request: Request):
    """Yalnızca yönetici hesabı için giriş ekranı."""
    return templates.TemplateResponse("admin_login.html", {"request": request, "error": None})


@app.post("/admin/login", response_class=HTMLResponse)
def admin_login_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = (
        db.query(User)
        .filter(User.email == email, User.password == password, User.is_admin == True)
        .first()
    )
    if not user:
        return templates.TemplateResponse(
            "admin_login.html",
            {"request": request, "error": "Geçersiz yönetici bilgileri."},
        )

    request.session["user_id"] = user.id
    return RedirectResponse(url="/admin/global", status_code=302)



@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)


@app.get("/upload", response_class=HTMLResponse)
def upload_get(request: Request):
    """Müşteri veriyi yüklerken login gerektirmeyen form ekranı."""
    return templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "user": None,
        },
    )



@app.post("/upload", response_class=HTMLResponse)
async def upload_post(
    request: Request,
    full_name: str = Form(...),
    company_name: str = Form(...),
    phone: str = Form(...),
    email: str = Form(...),
    sector: str = Form(""),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Login olmadan, müşteri bilgisi + CSV alıp analiz yapan endpoint."""
    content = await file.read()

    # CSV > Excel fallback
    try:
        df = pd.read_csv(io.BytesIO(content))
        file_type = "csv"
    except Exception:
        try:
            df = pd.read_excel(io.BytesIO(content))
            file_type = "excel"
        except Exception:
            return templates.TemplateResponse(
                "upload.html",
                {
                    "request": request,
                    "user": None,
                    "error": "Dosya okunamadı. Lütfen geçerli bir CSV/Excel dosyası yükleyin.",
                },
            )

    rows, cols = df.shape
    total_cells = int(rows * cols)
    total_missing = int(df.isna().sum().sum())
    quality_score = 100.0
    if total_cells > 0:
        quality_score = max(0.0, 100.0 - (total_missing / total_cells) * 100.0)

    top_missing_col = df.isna().sum().idxmax() if cols > 0 else None
    var_series = df.var(numeric_only=True)
    top_var_col = var_series.idxmax() if not var_series.empty else None

    domain_insights = ["Veri alani belirtilmemis, ozel KPI calismasi onerilir."]

    company_label = company_name or "Firma Belirtilmedi"

    # Upload kaydı (user_id yok, anonim müşteri yüklemesi)
    upload = Upload(
        user_id=None,
        file_name=file.filename,
        file_type=file_type,
        company=company_label,
        row_count=rows,
        col_count=cols,
        total_cells=total_cells,
        total_missing=total_missing,
        quality_score=quality_score,
        top_missing_col=top_missing_col,
        top_var_col=top_var_col,
        domain_insights="\n".join(domain_insights),
        contact_name=full_name,
        contact_phone=phone,
        contact_email=email,
        contact_sector=sector,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)

    # AI analiz
    ai_insights = ai_analyze_dataframe(df)
    ai_summary = ai_insights.get("summary", "")
    ai_risks = ai_insights.get("risks", "")
    ai_features = ai_insights.get("features", "")
    ai_models = ai_insights.get("ml_models", "")
    ai_recommendations = ai_insights.get("recommendations", "")

    upload.ai_summary = ai_summary
    upload.ai_risks = ai_risks
    upload.ai_features = ai_features
    upload.ai_models = ai_models
    upload.ai_recommendations = ai_recommendations
    db.commit()
    db.refresh(upload)

    # Grafikler
    charts_raw = generate_charts(df, upload_id=upload.id)
    chart_cards = build_chart_cards(charts_raw)

    # Cache
    ANALYSIS_CACHE[upload.id] = {
        "file_name": file.filename,
        "file_type": file_type,
        "row_count": rows,
        "col_count": cols,
        "total_cells": total_cells,
        "total_missing": total_missing,
        "quality_score": quality_score,
        "top_missing_col": top_missing_col,
        "top_var_col": top_var_col,
        "domain_insights": domain_insights,
        "charts": chart_cards,
        "company": company_label,
        "ai_summary": ai_summary,
        "ai_risks": ai_risks,
        "ai_features": ai_features,
        "ai_models": ai_models,
        "ai_recommendations": ai_recommendations,
        "contact_name": full_name,
        "contact_phone": phone,
        "contact_email": email,
        "contact_sector": sector,
    }

    request.session["last_upload_id"] = upload.id

    analysis_ctx = {
        "rows": rows,
        "cols": cols,
        "total_cells": total_cells,
        "total_missing": total_missing,
        "quality_score": quality_score,
        "top_missing_col": top_missing_col,
        "top_var_col": top_var_col,
        "domain_insights": domain_insights,
        "charts": chart_cards,
        "ai_summary": ai_summary,
        "ai_risks": ai_risks,
        "ai_features": ai_features,
        "ai_models": ai_models,
        "ai_recommendations": ai_recommendations,
    }

    return templates.TemplateResponse(
        "report.html",
        {
            "request": request,
            "user": None,
            "analysis": analysis_ctx,
            "charts": chart_cards,
            "ai_comment": ai_summary,
            "ai_report": ai_recommendations,
            "company": company_label,
            "file_name": file.filename,
            "file_type": file_type,
            "contact_name": full_name,
            "contact_phone": phone,
            "contact_email": email,
            "contact_sector": sector,
            "upload_id": upload.id, 
        },
    )
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    user = current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    uploads = (
        db.query(Upload)
        .filter(Upload.user_id == user.id)
        .order_by(Upload.created_at.desc())
        .all()
    )

    data = None
    charts: List[Dict[str, Any]] = []

    last_upload_id = request.session.get("last_upload_id")
    if last_upload_id and last_upload_id in ANALYSIS_CACHE:
        cached = ANALYSIS_CACHE[last_upload_id]
        data = {
            "quality_score": cached["quality_score"],
            "total_rows": cached["row_count"],
            "total_columns": cached["col_count"],
            "total_missing": cached["total_missing"],
            "ai_summary": cached.get("ai_summary", ""),
            "ai_details": cached.get("ai_recommendations", ""),
        }
        charts = cached.get("charts", [])

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": user,
            "uploads": uploads,
            "data": data,
            "charts": charts,
        },
    )


@app.get("/reports", response_class=HTMLResponse)
def reports(request: Request, db: Session = Depends(get_db)):
    user = current_user(request, db)
    if not user:
        return RedirectResponse(url="/login", status_code=302)

    uploads = (
        db.query(Upload)
        .filter(Upload.user_id == user.id)
        .order_by(Upload.created_at.desc())
        .all()
    )

    return templates.TemplateResponse(
        "reports.html",
        {
            "request": request,
            "user": user,
            "uploads": uploads,
        },
    )


@app.get("/admin", response_class=HTMLResponse)
def admin_redirect():
    return RedirectResponse(url="/admin/global", status_code=302)


@app.get("/admin/global", response_class=HTMLResponse)
def admin_global(request: Request, db: Session = Depends(get_db)):
    user = current_user(request, db)
    if not user or not user.is_admin:
        return RedirectResponse(url="/admin/login", status_code=302)

    total_users = db.query(User).count()
    total_uploads = db.query(Upload).count()
    last_uploads = (
        db.query(Upload)
        .order_by(Upload.created_at.desc())
        .limit(10)
        .all()
    )

    return templates.TemplateResponse(
        "admin_global.html",
        {
            "request": request,
            "user": user,
            "total_users": total_users,
            "total_uploads": total_uploads,
            "last_uploads": last_uploads,
        },
    )



@app.get("/download_pdf/{upload_id}")
def download_pdf(upload_id: int, db: Session = Depends(get_db)):
    """
    ANALYSIS_CACHE veya DB'deki AI sonuçlarını ve grafik dosyalarını kullanarak PDF raporu indir.
    """
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload:
        raise HTTPException(status_code=404, detail="Rapor bulunamadı.")

    # Önce cache, yoksa DB'den AI alanlarını doldur
    cached = ANALYSIS_CACHE.get(upload_id)
    if cached:
        summary = cached.get("ai_summary", upload.ai_summary or "")
        risks = cached.get("ai_risks", upload.ai_risks or "")
        features = cached.get("ai_features", upload.ai_features or "")
        models = cached.get("ai_models", upload.ai_models or "")
        recs = cached.get("ai_recommendations", upload.ai_recommendations or "")
    else:
        summary = upload.ai_summary or ""
        risks = upload.ai_risks or ""
        features = upload.ai_features or ""
        models = upload.ai_models or ""
        recs = upload.ai_recommendations or ""

    # Cache içindeki grafik kartlarından dosya yollarını hazırla
    chart_files: List[str] = []
    if cached:
        chart_cards = cached.get("charts") or []
        for ch in chart_cards:
            url = ch.get("url") if isinstance(ch, dict) else None
            if not url:
                continue
            fname = os.path.basename(url)
            fpath = os.path.join(CHART_DIR, fname)
            if os.path.exists(fpath):
                chart_files.append(fpath)

    pdf_path = os.path.join(REPORT_DIR, f"rapor_{upload_id}.pdf")
    generate_pdf_report(
        pdf_path,
        summary=summary,
        risks=risks,
        features=features,
        models=models,
        recommendations=recs,
        chart_files=chart_files,
    )

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"veri_raporu_{upload_id}.pdf",
    )

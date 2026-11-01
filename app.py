import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# ----------------------------------------------------------------------------
# Konfigurasi halaman
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Klasifikasi Organisme Laut — CNN",
    page_icon="🐠",
    layout="wide",
)

PRIMARY = "#0F7C8A"   # warna utama (sama dengan notebook)
MUTED = "#B8A98A"
ACCENT = "#D9534F"
IMG_SIZE = (160, 160)  # sama dengan IMG_SIZE di notebook

ID_NAMES = {
    "other": "Lainnya", "fish": "Ikan", "seal": "Anjing laut", "coral": "Terumbu karang",
    "crab": "Kepiting", "shark": "Hiu", "turtle": "Penyu", "octopus": "Gurita",
    "starfish": "Bintang laut", "jellyfish": "Ubur-ubur",
}
EMOJI = {
    "other": "🌊", "fish": "🐟", "seal": "🦭", "coral": "🪸", "crab": "🦀",
    "shark": "🦈", "turtle": "🐢", "octopus": "🐙", "starfish": "⭐", "jellyfish": "🪼",
}

# Cadangan: nilai dari output notebook, dipakai bila file artefak belum ada.
FALLBACK_CLASSES = ["other", "fish", "seal", "coral", "crab", "shark", "turtle",
                    "octopus", "starfish", "jellyfish"]
FALLBACK_SPLIT = pd.DataFrame(
    {"train": [320, 210, 54, 50, 26, 50, 30, 27, 27, 26],
     "validation": [40, 26, 7, 6, 4, 6, 4, 3, 3, 4],
     "test": [40, 27, 7, 6, 3, 6, 4, 3, 4, 3]},
    index=FALLBACK_CLASSES,
)
FALLBACK_SCAN = {"n_files": 7990, "n_corrupt": 8, "n_duplicates": 6956, "n_valid": 1026}
FALLBACK_METRICS = {
    "accuracy": 0.8058,
    "macro": {"precision": 0.8012, "recall": 0.8694, "f1": 0.8179},
    "weighted": {"precision": 0.8270, "recall": 0.8058, "f1": 0.8075},
    "per_class": {
        "other": {"precision": 0.844, "recall": 0.675, "f1": 0.750, "support": 40},
        "fish": {"precision": 0.821, "recall": 0.852, "f1": 0.836, "support": 27},
        "seal": {"precision": 0.875, "recall": 1.0, "f1": 0.933, "support": 7},
        "coral": {"precision": 0.714, "recall": 0.833, "f1": 0.769, "support": 6},
        "crab": {"precision": 1.0, "recall": 0.667, "f1": 0.800, "support": 3},
        "shark": {"precision": 0.857, "recall": 1.0, "f1": 0.923, "support": 6},
        "turtle": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 4},
        "octopus": {"precision": 0.5, "recall": 1.0, "f1": 0.667, "support": 3},
        "starfish": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 4},
        "jellyfish": {"precision": 0.4, "recall": 0.667, "f1": 0.5, "support": 3},
    },
}


# ----------------------------------------------------------------------------
# Pemuatan data (di-cache)
# ----------------------------------------------------------------------------
def _read_json(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_artifacts(out_dir: str):
    d = Path(out_dir)
    art = {"missing": []}

    classes = _read_json(d / "class_names.json")
    if not classes:
        art["missing"].append("class_names.json")
        classes = FALLBACK_CLASSES
    art["classes"] = classes

    # split per kelas
    try:
        split = pd.read_csv(d / "split_counts.csv", index_col=0)
        split = split[["train", "validation", "test"]]
    except Exception:
        art["missing"].append("split_counts.csv")
        split = FALLBACK_SPLIT.copy()
    art["split"] = split.reindex(classes).fillna(0).astype(int)

    scan = _read_json(d / "scan_info.json")
    if not scan:
        art["missing"].append("scan_info.json")
        scan = FALLBACK_SCAN
    art["scan"] = scan

    metrics = _read_json(d / "metrics.json")
    if not metrics:
        art["missing"].append("metrics.json")
        metrics = FALLBACK_METRICS
    art["metrics"] = metrics

    hist = _read_json(d / "history.json")
    if not hist:
        art["missing"].append("history.json")
    art["history"] = hist

    cm = None
    try:
        cm = np.load(d / "confusion_matrix.npy")
    except Exception:
        art["missing"].append("confusion_matrix.npy")
    art["cm"] = cm

    art["config"] = _read_json(d / "config.json") or {}
    return art


@st.cache_resource(show_spinner="Memuat model CNN…")
def load_model(out_dir: str):
    """Muat model Keras. Mengembalikan (model, pesan_error)."""
    try:
        import tensorflow as tf  # impor di sini agar dashboard tetap jalan tanpa TF
    except Exception as exc:
        return None, f"TensorFlow tidak terpasang ({exc}). Jalankan: pip install tensorflow"
    for name in ("final_oceanic_model.keras", "oceanic_best.keras"):
        p = Path(out_dir) / name
        if p.exists():
            try:
                return tf.keras.models.load_model(p, compile=False), None
            except Exception as exc:
                return None, f"Gagal memuat {p.name}: {exc}"
    return None, (f"File model tidak ditemukan di '{out_dir}'. "
                  "Pastikan final_oceanic_model.keras ada di folder output.")


def preprocess(img: Image.Image) -> np.ndarray:
    """Sama seperti notebook: EXIF transpose -> RGB -> resize LANCZOS 160x160 -> /255.
    (Rescaling ke [-1,1] untuk MobileNetV2 sudah ada DI DALAM model.)"""
    im = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE, Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.float32)[None] / 255.0


def label(c: str) -> str:
    return f"{EMOJI.get(c, '')} {c} ({ID_NAMES.get(c, c)})".strip()


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
default_out = os.environ.get("OUT_DIR", str(Path(__file__).parent / "output"))
with st.sidebar:
    st.divider()
    st.markdown(
        "**Navigasi**\n"
        "1. Ringkasan dataset\n2. Distribusi kelas\n3. Hasil pelatihan\n"
        "4. Confusion matrix\n5. Evaluasi model\n6. Uji prediksi"
    )
    st.divider()
    st.caption("Model: MobileNetV2 (transfer learning + fine-tuning)\n\n"
               "Label dataset dibuat otomatis dengan CLIP (pseudo-label).")

out_dir = default_out

art = load_artifacts(out_dir)
classes = art["classes"]
split = art["split"]
metrics = art["metrics"]
history = art["history"]
cm = art["cm"]
cfg = art["config"]
counts = split.sum(axis=1)
n_total = int(counts.sum())

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🐠 Klasifikasi Citra Organisme Laut dengan CNN")
st.markdown(
    "Dashboard ini menampilkan hasil eksperimen **Convolutional Neural Network (CNN)** "
    "berbasis **MobileNetV2** untuk mengenali **10 kelas objek laut** pada citra: "
    "ikan, anjing laut, terumbu karang, kepiting, hiu, penyu, gurita, bintang laut, "
    "ubur-ubur, dan kelas *other* (objek laut lain / bukan target). "
    "Alur eksperimen: inspeksi data → pelabelan otomatis (CLIP) → EDA → preprocessing → "
    "split 80/10/10 → pelatihan 2 fase (head + fine-tuning) → evaluasi pada data uji."
)
if art["missing"]:
    st.warning(
        "File artefak belum ditemukan di folder output: **" + ", ".join(art["missing"]) +
        "**. Beberapa bagian memakai nilai cadangan dari notebook atau tidak ditampilkan."
    )

tabs = st.tabs([
    "📦 Ringkasan Dataset", "📊 Distribusi Kelas", "📈 Hasil Pelatihan",
    "🧩 Confusion Matrix", "🎯 Evaluasi Model", "🔍 Uji Prediksi",
])

# ----------------------------------------------------------------------------
# 1. Ringkasan dataset
# ----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Ringkasan Dataset")
    tr, va, te = (int(split[c].sum()) for c in ("train", "validation", "test"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah kelas", len(classes))
    c2.metric("Gambar valid dipakai", f"{n_total:,}")
    c3.metric("File mentah ditemukan", f"{art['scan'].get('n_files', '-'):,}")
    c4.metric("Resolusi input model", f"{IMG_SIZE[0]}×{IMG_SIZE[1]}")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Proporsi data latih – validasi – uji**")
        prop = pd.DataFrame({
            "Subset": ["Latih (train)", "Validasi", "Uji (test)"],
            "Jumlah gambar": [tr, va, te],
            "Proporsi": [f"{tr/n_total:.1%}", f"{va/n_total:.1%}", f"{te/n_total:.1%}"],
        })
        st.dataframe(prop, hide_index=True)

        fig, ax = plt.subplots(figsize=(5, 3.2))
        wedges, _, autot = ax.pie(
            [tr, va, te], labels=["Latih", "Validasi", "Uji"], autopct="%1.1f%%",
            colors=[PRIMARY, MUTED, ACCENT], startangle=90,
            wedgeprops=dict(width=0.45, edgecolor="white"), textprops={"fontsize": 9})
        for t in autot:
            t.set_color("white"); t.set_fontsize(8)
        ax.set_title("Pembagian data (stratified)")
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        st.markdown("**Pembersihan data**")
        s = art["scan"]
        clean = pd.DataFrame({
            "Tahap": ["File gambar ditemukan", "File korup dibuang",
                      "Duplikat persis (MD5) dibuang", "Gambar valid dipakai"],
            "Jumlah": [s.get("n_files"), s.get("n_corrupt"), s.get("n_duplicates"), s.get("n_valid")],
        })
        st.dataframe(clean, hide_index=True)
        st.markdown("**Rincian per kelas**")
        tbl = split.copy()
        tbl["total"] = tbl.sum(axis=1)
        tbl.index = [f"{c} ({ID_NAMES.get(c, c)})" for c in tbl.index]
        st.dataframe(tbl)

    dup_pct = s.get("n_duplicates", 0) / max(s.get("n_files", 1), 1)
    st.markdown("#### 📝 Analisis")
    st.write(
        f"Dataset akhir berisi **{n_total:,} gambar** dalam **{len(classes)} kelas**, dibagi "
        f"**{tr/n_total:.0%} latih ({tr}), {va/n_total:.0%} validasi ({va}), dan {te/n_total:.0%} "
        f"uji ({te})** dengan stratifikasi sehingga setiap kelas hadir di ketiga subset. "
        f"Dari {s.get('n_files', 0):,} file mentah, sekitar **{dup_pct:.0%}** adalah duplikat persis "
        f"dan {s.get('n_corrupt', 0)} file korup, sehingga data bersih jauh lebih kecil dari data "
        f"mentah. Karena data uji hanya **{te} gambar**, satu kesalahan prediksi saja menggeser "
        f"akurasi sekitar {1/te:.1%}, hasil evaluasi perlu dibaca dengan kehati-hatian."
    )

# ----------------------------------------------------------------------------
# 2. Distribusi kelas
# ----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Distribusi Data per Kelas")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    bars = ax.bar(counts.index, counts.values,
                  color=[MUTED if c == "other" else PRIMARY for c in counts.index])
    ax.axhline(counts.mean(), color=ACCENT, ls="--", label=f"Rata-rata = {counts.mean():.0f}")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, str(int(v)), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Jumlah gambar")
    ax.set_title("Distribusi gambar per kelas (label CLIP)")
    ax.legend()
    plt.xticks(rotation=30)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Lihat distribusi per subset (train / validation / test)"):
        fig, ax = plt.subplots(figsize=(10, 4))
        split.plot.bar(stacked=True, ax=ax, color=[PRIMARY, MUTED, ACCENT])
        ax.set_ylabel("Jumlah gambar"); ax.set_xlabel("")
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    imb = counts.max() / max(counts.min(), 1)
    level = ("relatif seimbang" if imb < 1.5 else
             "tidak seimbang sedang" if imb < 3 else "sangat tidak seimbang")
    top2 = counts.sort_values(ascending=False).head(2)
    st.markdown("#### 📝 Analisis")
    st.write(
        f"Kelas terbanyak adalah **{counts.idxmax()} ({counts.max()} gambar, "
        f"{counts.max()/n_total:.0%} dari total)** dan tersedikit **{counts.idxmin()} "
        f"({counts.min()} gambar)**, rasio **{imb:.1f}×** → data **{level}**. "
        f"Dua kelas teratas ({', '.join(top2.index)}) menyumbang "
        f"{top2.sum()/n_total:.0%} data, sehingga model berisiko cenderung memprediksi kelas "
        f"mayoritas. Kelas *other* besar karena memuat banyak jenis objek (paus, lumba-lumba, "
        f"alga, penyelam, dll.) dan gambar ber-confidence CLIP rendah. Untuk mengimbangi, "
        f"pelatihan memakai **class weight** (kelas kecil diberi bobot lebih besar) dan augmentasi "
        f"data. Kelas minoritas (≈{counts.min()}–{int(counts.sort_values().iloc[3])} gambar) "
        f"hanya punya 3–4 contoh di data uji, sehingga metriknya mudah berfluktuasi."
    )

# ----------------------------------------------------------------------------
# 3. Hasil pelatihan
# ----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Grafik Hasil Pelatihan (Akurasi & Loss)")
    if not history:
        st.error("history.json tidak ditemukan di folder output, grafik pelatihan tidak dapat ditampilkan.")
    else:
        split_ep = int(history.get("split_epoch", 0))
        n_ep = len(history["loss"])
        epochs = np.arange(1, n_ep + 1)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
        for a, (m, title) in zip(axes, [("accuracy", "Accuracy"), ("loss", "Loss")]):
            a.plot(epochs, history[m], "o-", color=PRIMARY, label=f"Train {title}")
            a.plot(epochs, history[f"val_{m}"], "s-", color=ACCENT, label=f"Val {title}")
            if 0 < split_ep < n_ep:
                a.axvline(split_ep + 0.5, color="gray", ls="--", label="Mulai fine-tuning")
            a.set_title(f"Training vs Validation ({title})")
            a.set_xlabel("Epoch"); a.set_ylabel(title)
            a.grid(alpha=.3); a.legend()
            a.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        va_loss = np.array(history["val_loss"])
        best = int(va_loss.argmin())
        tr_acc, va_acc = history["accuracy"][-1], history["val_accuracy"][-1]
        gap = tr_acc - va_acc
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total epoch dijalankan", n_ep)
        c2.metric("Epoch terbaik (val loss)", best + 1, f"{va_loss[best]:.3f}")
        c3.metric("Train acc (akhir)", f"{tr_acc:.1%}")
        c4.metric("Val acc (akhir)", f"{va_acc:.1%}", f"{-gap:+.1%} vs train", delta_color="off")

        if gap > 0.10:
            fit = (f"Selisih akurasi train–validasi {gap:.1%} menunjukkan **indikasi overfitting** "
                   "(model lebih hafal data latih dibanding generalisasi).")
        elif gap < -0.05:
            fit = ("Akurasi validasi lebih tinggi daripada train; hal ini wajar karena augmentasi & "
                   "dropout aktif saat training namun nonaktif saat validasi.")
        else:
            fit = (f"Selisih akurasi train–validasi hanya {abs(gap):.1%} → model **cukup "
                   "seimbang (tidak overfit parah)**.")
        ft = ""
        if 0 < split_ep < n_ep:
            a_before = history["val_accuracy"][split_ep - 1]
            a_after = max(history["val_accuracy"][split_ep:])
            ft = (f" Fase 1 (melatih classifier head) berhenti di epoch {split_ep} dengan val acc "
                  f"{a_before:.1%}; fine-tuning 30 layer terakhir mencapai val acc tertinggi "
                  f"{a_after:.1%} ({a_after - a_before:+.1%}).")
        st.markdown("#### 📝 Analisis")
        st.write(
            f"Loss validasi terendah (**{va_loss[best]:.3f}**) tercapai pada **epoch {best+1}**, "
            f"dan bobot terbaik itulah yang disimpan (EarlyStopping + ModelCheckpoint). "
            f"{fit}{ft} Kurva yang berfluktuasi pada validasi wajar karena set validasi kecil "
            f"({int(split['validation'].sum())} gambar)."
        )

# ----------------------------------------------------------------------------
# 4. Confusion matrix
# ----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Confusion Matrix (Data Uji)")
    if cm is None:
        st.error("confusion_matrix.npy tidak ditemukan di folder output.")
    else:
        n = len(classes)
        cm_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(8.5, 7))
        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, label="Proporsi per kelas asli")
        ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(classes)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if cm_pct[i, j] > .5 else "black")
        ax.set_xlabel("Prediksi"); ax.set_ylabel("Kelas sebenarnya (pseudo-label CLIP)")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        _, mid, _ = st.columns([1, 3, 1])
        with mid:
            st.pyplot(fig)
        plt.close(fig)

        pairs = sorted(
            [(classes[i], classes[j], int(cm[i, j])) for i in range(n) for j in range(n)
             if i != j and cm[i, j] > 0], key=lambda t: -t[2])
        correct = int(np.trace(cm)); tot = int(cm.sum())
        recall_c = pd.Series(np.diag(cm) / np.maximum(cm.sum(axis=1), 1), index=classes)
        prec_c = pd.Series(np.diag(cm) / np.maximum(cm.sum(axis=0), 1), index=classes)
        st.markdown("**Kesalahan terbanyak (asli → prediksi)**")
        if pairs:
            st.dataframe(pd.DataFrame(pairs[:6], columns=["Kelas asli", "Diprediksi sebagai", "Jumlah"]),
                         hide_index=True)
        st.markdown("#### 📝 Analisis")
        txt = (f"Dari **{tot}** gambar uji, **{correct}** diprediksi benar (diagonal utama) dan "
               f"**{tot - correct}** salah. ")
        if pairs:
            a_, b_, k_ = pairs[0]
            txt += (f"Kesalahan paling sering adalah **{a_} → {b_} ({k_} gambar)**. ")
        wr = recall_c.idxmin()
        wp = prec_c.idxmin()
        txt += (f"Recall terendah ada pada **{wr}** ({recall_c.min():.0%}) dan precision terendah "
                f"pada **{wp}** ({prec_c.min():.0%}), artinya banyak gambar kelas lain "
                f"'tersedot' ke kelas {wp}. ")
        if "other" in classes:
            io = classes.index("other")
            leak = int(cm[io].sum() - cm[io, io])
            txt += (f"Kelas *other* (heterogen: berisi aneka objek laut) sering salah terbaca "
                    f"sebagai kelas spesifik — **{leak} dari {int(cm[io].sum())}** gambar *other* "
                    f"salah prediksi. Ini wajar karena batas visual antara *other* dan organisme "
                    f"target tidak tegas, ditambah label *other* sebagian berasal dari ambang "
                    f"confidence CLIP.")
        st.write(txt)

# ----------------------------------------------------------------------------
# 5. Evaluasi
# ----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Hasil Evaluasi Model (Data Uji)")
    m, w = metrics["macro"], metrics["weighted"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    c2.metric("Precision (macro)", f"{m['precision']:.2%}", f"weighted {w['precision']:.2%}", delta_color="off")
    c3.metric("Recall (macro)", f"{m['recall']:.2%}", f"weighted {w['recall']:.2%}", delta_color="off")
    c4.metric("F1-score (macro)", f"{m['f1']:.2%}", f"weighted {w['f1']:.2%}", delta_color="off")

    pc = pd.DataFrame(metrics["per_class"]).T.reindex(classes)
    pc["support"] = pc["support"].astype(int)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(classes)); wd = 0.27
    ax.bar(x - wd, pc["precision"], wd, label="Precision", color=PRIMARY)
    ax.bar(x, pc["recall"], wd, label="Recall", color=MUTED)
    ax.bar(x + wd, pc["f1"], wd, label="F1-score", color=ACCENT)
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=30)
    ax.set_ylim(0, 1.08); ax.set_title("Precision, Recall, F1-score per kelas")
    ax.legend(ncol=3, loc="lower right"); ax.grid(axis="y", alpha=.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    show = pc.rename(columns={"precision": "Precision", "recall": "Recall",
                              "f1": "F1-score", "support": "Support"})
    st.dataframe(show.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}"})
                 .background_gradient(subset=["F1-score"], cmap="Blues", vmin=0, vmax=1))

    f1s = pc["f1"]
    low = pc[pc["f1"] < 0.7]
    gap = abs(w["f1"] - m["f1"])
    st.markdown("#### 📝 Analisis")
    st.write(
        f"Model mencapai **akurasi {metrics['accuracy']:.1%}** pada data uji; F1 macro "
        f"**{m['f1']:.3f}** vs weighted **{w['f1']:.3f}** "
        f"({'selisih kecil → kinerja relatif merata antar kelas' if gap < .03 else 'selisih besar → kelas kecil tertinggal'}). "
        f"Kelas terbaik: **{f1s.idxmax()}** (F1 {f1s.max():.2f}); terlemah: **{f1s.idxmin()}** "
        f"(F1 {f1s.min():.2f}). "
        + (f"Kelas dengan F1 < 0.70: {', '.join(low.index)} — umumnya karena contoh uji sangat sedikit "
           f"(support {int(low['support'].min())}–{int(low['support'].max())}) dan "
           f"penampilan visual yang mirip kelas lain. " if len(low) else "")
        + f"Recall macro ({m['recall']:.2f}) lebih tinggi dari precision macro ({m['precision']:.2f}): "
          "berkat class weight, model lebih 'berani' memprediksi kelas minoritas sehingga kelas kecil "
          "jarang terlewat tetapi kadang salah tebak (precision turun). Mengingat support kelas kecil "
          "hanya 3–7 gambar dan label berasal dari CLIP, angka per kelas sebaiknya dianggap indikatif."
    )

# ----------------------------------------------------------------------------
# 6. Uji prediksi
# ----------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Uji Prediksi Gambar Baru (Real-time)")
    st.write("Unggah satu atau beberapa gambar (JPG/PNG/WEBP), atau ambil foto dengan kamera. "
             "Prediksi langsung muncul begitu gambar dipilih.")

    model, err = load_model(out_dir)
    if err:
        st.error(err)

    src = st.radio("Sumber gambar", ["Upload file", "Kamera"], horizontal=True)
    files = []
    if src == "Upload file":
        files = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png", "webp", "bmp", "gif"],
                                 accept_multiple_files=True) or []
    else:
        cam = st.camera_input("Ambil foto")
        files = [cam] if cam else []

    if files and model is not None:
        out_names = [c for c in classes]
        # urutan output model = urutan CLASS_NAMES di notebook (class_names.json)
        if model.output_shape[-1] != len(out_names):
            st.error(f"Jumlah output model ({model.output_shape[-1]}) ≠ jumlah kelas "
                     f"({len(out_names)}). Periksa class_names.json.")
            st.stop()
        summary = []
        for f in files:
            try:
                img = Image.open(f)
                img.load()
            except Exception as exc:
                st.error(f"{getattr(f, 'name', 'gambar')}: tidak dapat dibaca ({exc})")
                continue
            probs = np.asarray(model(preprocess(img), training=False))[0]
            order = np.argsort(-probs)
            top, conf = out_names[order[0]], float(probs[order[0]])
            top2, conf2 = out_names[order[1]], float(probs[order[1]])
            summary.append({"File": getattr(f, "name", "kamera"), "Prediksi": top,
                            "Nama (ID)": ID_NAMES.get(top, top), "Confidence": f"{conf:.1%}"})

            st.divider()
            col_img, col_res = st.columns([1, 1.4])
            with col_img:
                st.image(ImageOps.exif_transpose(img).convert("RGB"), width="stretch",
                         caption=getattr(f, "name", "Foto kamera"))
            with col_res:
                st.markdown(f"### {EMOJI.get(top, '')} Prediksi: **{top.upper()}** — {ID_NAMES.get(top, top)}")
                st.progress(min(conf, 1.0), text=f"Confidence {conf:.1%}")

                fig, ax = plt.subplots(figsize=(6, 3.6))
                idx = order[::-1]
                ax.barh([out_names[i] for i in idx], probs[idx],
                        color=[PRIMARY if i == order[0] else MUTED for i in idx])
                ax.set_xlim(0, 1); ax.set_xlabel("Probabilitas")
                ax.set_title("Probabilitas per kelas")
                for y, i in enumerate(idx):
                    ax.text(probs[i] + 0.01, y, f"{probs[i]:.1%}", va="center", fontsize=8)
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # analisis otomatis
            if conf >= 0.85:
                lvl = "sangat yakin"
            elif conf >= 0.60:
                lvl = "cukup yakin"
            else:
                lvl = "kurang yakin"
            txt = (f"Model **{lvl}** bahwa gambar ini adalah **{top} ({ID_NAMES.get(top, top)})** "
                   f"dengan probabilitas {conf:.1%}. Kandidat kedua: **{top2}** ({conf2:.1%}). ")
            if conf - conf2 < 0.20:
                txt += ("Selisih dengan kandidat kedua kecil, sehingga hasil ambigu; coba foto yang "
                        "lebih jelas/objek lebih dominan. ")
            if top == "other":
                txt += ("Kelas *other* berarti objek tidak termasuk 9 organisme target (atau tidak "
                        "dikenali dengan yakin). ")
            if conf < 0.60:
                txt += ("Confidence rendah — hasil ini sebaiknya tidak dijadikan kesimpulan. ")
            txt += ("Ingat: model hanya mengenal 10 kelas ini dan dilatih pada ±820 gambar, jadi "
                    "gambar di luar domain (mis. manusia, hewan darat) tetap akan dipaksa masuk ke "
                    "salah satu kelas.")
            st.markdown("#### 📝 Analisis")
            st.write(txt)

        if len(summary) > 1:
            st.divider()
            st.markdown("**Rekap semua prediksi**")
            st.dataframe(pd.DataFrame(summary), hide_index=True)
    elif not files:
        st.caption("Belum ada gambar. Unggah gambar untuk melihat prediksi.")

st.divider()
st.caption("Dashboard CNN • MobileNetV2 + fine-tuning • label dataset: CLIP zero-shot (pseudo-label)")
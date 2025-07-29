import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import ast
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import re
from collections import Counter
import io

# Import SMOTE dari imblearn
from imblearn.over_sampling import SMOTE


# Coba unduh data NLTK yang dibutuhkan
try:
    nltk.data.find('corpora/stopwords')
except Exception:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except Exception:
    nltk.download('punkt')


# Atur konfigurasi halaman Streamlit
st.set_page_config(layout="wide", page_title="Dashboard Analisis Sentimen")

# --- Fungsi untuk WordCloud per Sentimen (Disesuaikan untuk mengembalikan teks juga) ---
@st.cache_data
def generate_sentiment_wordclouds(df, text_column='teks', sentiment_column='label'):
    """
    Menghasilkan objek WordCloud untuk sentimen positif dan negatif.
    Juga mengembalikan teks gabungan untuk analisis kata populer.
    """
    if text_column not in df.columns or sentiment_column not in df.columns:
        st.error(f"Kolom '{text_column}' atau '{sentiment_column}' tidak ditemukan di DataFrame untuk WordCloud Sentimen.")
        return {}, "", ""

    df_sentiment = df.copy()

    df_sentiment['processed_text_for_wc'] = df_sentiment[text_column].apply(lambda x:
                                                                            " ".join(map(str, x)) if isinstance(x, (list, np.ndarray)) else
                                                                            (str(x) if not pd.isna(x) else "")
                                                                            )
    
    df_sentiment = df_sentiment.dropna(subset=[sentiment_column])

    label_mapping = {0: 'Negatif', 1: 'Positif'}
    df_sentiment[sentiment_column] = df_sentiment[sentiment_column].map(label_mapping)

    positive_texts = " ".join(df_sentiment[df_sentiment[sentiment_column] == 'Positif']['processed_text_for_wc'].tolist())
    negative_texts = " ".join(df_sentiment[df_sentiment[sentiment_column] == 'Negatif']['processed_text_for_wc'].tolist())

    wordclouds = {}

    if positive_texts:
        wordcloud_positif = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='Blues',
            min_font_size=10,
            collocations=False
        ).generate(positive_texts)
        wordclouds['Positif'] = wordcloud_positif
    else:
        pass

    if negative_texts:
        wordcloud_negatif = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='Reds',
            min_font_size=10,
            collocations=False
        ).generate(negative_texts)
        wordclouds['Negatif'] = wordcloud_negatif
    else:
        pass

    return wordclouds, positive_texts, negative_texts


# --- Variabel Global dan Pemuatan Data (Disimpan dalam Cache untuk Performa) ---
@st.cache_data
def load_and_split_data(file_path):
    """
    Memuat data dari CSV, mengidentifikasi kolom embedding, dan membagi
    data ke dalam set pelatihan dan pengujian.
    """
    if not os.path.exists(file_path):
        st.error(f"Error: File '{file_path}' tidak ditemukan. Pastikan file berada di direktori yang benar.")
        st.stop()

    try:
        df_data = pd.read_csv(file_path)

        if 'teks' in df_data.columns:
            def process_text_entry_for_load(x):
                if pd.isna(x):
                    return []
                if isinstance(x, str):
                    x_stripped = x.strip()
                    if x_stripped.startswith('[') and x_stripped.endswith(']'):
                        try:
                            evaluated = ast.literal_eval(x_stripped)
                            if isinstance(evaluated, (list, np.ndarray)):
                                return list(evaluated)
                            return [str(evaluated)]
                        except (ValueError, SyntaxError):
                            pass
                    return [x]
                if isinstance(x, np.ndarray):
                    return x.tolist()
                if isinstance(x, list):
                    return x
                return [str(x)]

            df_data['teks'] = df_data['teks'].apply(process_text_entry_for_load)

        if 'tanggal' in df_data.columns:
            df_data['tanggal_dt'] = pd.to_datetime(df_data['tanggal'], errors='coerce')
            df_data['year'] = df_data['tanggal_dt'].dt.year
            df_data = df_data.dropna(subset=['year'])
            df_data['year'] = df_data['year'].astype(int)
        else:
            df_data['year'] = 2024

        embedding_cols = [col for col in df_data.columns if col.startswith('embedding_')]
        if not embedding_cols:
            st.error("Error: Kolom embeddings tidak ditemukan di DataFrame Anda. Pastikan nama kolom dimulai dengan 'embedding_'.")
            st.stop()

        if 'label' not in df_data.columns:
            st.error("Error: Kolom 'label' tidak ditemukan. Pastikan file input memiliki kolom label sentimen.")
            st.stop()

        for col in embedding_cols:
            df_data[col] = pd.to_numeric(df_data[col], errors='coerce')
        
        # st.info(f"Jumlah baris awal: {len(df_data)}") # Dihapus

        nan_in_embeddings_before = df_data[embedding_cols].isnull().any(axis=1).sum()
        inf_in_embeddings_before = np.isinf(df_data[embedding_cols].values).any().sum()
        nan_in_label_before = df_data['label'].isnull().sum()
        
        if nan_in_embeddings_before > 0 or inf_in_embeddings_before > 0 or nan_in_label_before > 0:
            # st.warning(f"Sebelum dropna: Ditemukan {nan_in_embeddings_before} baris dengan NaN di embedding, {inf_in_embeddings_before} baris dengan Inf di embedding, dan {nan_in_label_before} baris dengan NaN di 'label'.") # Dihapus
            pass

        df_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        initial_rows = len(df_data)
        df_data.dropna(subset=embedding_cols + ['label'], inplace=True)
        rows_after_na_drop = len(df_data)
        
        if initial_rows != rows_after_na_drop:
            # st.warning(f"Dihapus {initial_rows - rows_after_na_drop} baris karena mengandung nilai yang hilang (NaN/Inf) di kolom embedding atau label.") # Dihapus
            pass
        
        # st.info(f"Jumlah baris setelah penghapusan NaN/Inf: {len(df_data)}") # Dihapus
        
        if len(df_data) == 0:
            st.error("Setelah membersihkan data, tidak ada baris yang tersisa. Pastikan file CSV Anda memiliki data yang valid dan cukup.")
            st.stop()

        X_embeddings = df_data[embedding_cols].values.astype(np.float64)
        y_labels = df_data['label'].values.astype(int)

        if np.isnan(X_embeddings).any():
            st.error("FINAL CHECK: Masih ada nilai NaN di X_embeddings setelah semua pembersihan. Ini masalah serius!")
            st.stop()
        if np.isinf(X_embeddings).any():
            st.error("FINAL CHECK: Masih ada nilai Inf di X_embeddings setelah semua pembersihan. Ini masalah serius!")
            st.stop()
        if X_embeddings.ndim != 2:
            st.error(f"FINAL CHECK: X_embeddings harus 2D, tapi dimensinya adalah {X_embeddings.ndim}. Bentuk: {X_embeddings.shape}")
            st.stop()
        if y_labels.ndim != 1:
            st.error(f"FINAL CHECK: y_labels harus 1D, tapi dimensinya adalah {y_labels.ndim}. Bentuk: {y_labels.shape}")
            st.stop()
        
        # st.info(f"Bentuk X_embeddings final: {X_embeddings.shape}") # Dihapus
        # st.info(f"Bentuk y_labels final: {y_labels.shape}") # Dihapus

        test_size = 0.2
        random_seed = 42

        X_train, X_test, y_train, y_test = train_test_split(
            X_embeddings, y_labels, test_size=test_size, random_state=random_seed, stratify=y_labels
        )
        
        # st.info(f"Bentuk X_train setelah split: {X_train.shape}") # Dihapus
        # st.info(f"Bentuk y_train setelah split: {y_train.shape}") # Dihapus

        train_idx, test_idx, _, _ = train_test_split(
            df_data.index, df_data['label'], test_size=test_size, random_state=random_seed, stratify=df_data['label']
        )
        
        df_train_display = df_data.loc[train_idx]
        df_test_display = df_data.loc[test_idx]

        return df_data, X_train, X_test, y_train, y_test, embedding_cols, y_labels, df_train_display, df_test_display

    except Exception as e:
        st.error(f"Terjadi kesalahan SAAT PEMUATAN atau PEMBAGIAN data: {e}")
        st.exception(e)
        st.stop()

# Sidebar untuk Konfigurasi Dataset
st.sidebar.header("Konfigurasi Dataset")
default_file_path = "korupsi_labeled_and_embedded.csv"
input_combined_file = st.sidebar.text_input("Path File Data CSV", value=default_file_path)

# Muat data sekali dan simpan dalam cache
df_data, X_train, X_test, y_train, y_test, embedding_cols, y_labels_original, df_train_display, df_test_display = load_and_split_data(input_combined_file)

# --- PENYEIMBANGAN DATA DENGAN SMOTE (Diproses setelah load_and_split_data) ---
@st.cache_data
def apply_smote(X_train_data, y_train_data):
    # st.info(f"START SMOTE: Menerapkan SMOTE pada data dengan bentuk X_train_data: {X_train_data.shape}, y_train_data: {y_train_data.shape}") # Dihapus
    
    if np.isnan(X_train_data).any():
        st.error("SMOTE PRE-CHECK: NaN terdeteksi di X_train_data tepat sebelum SMOTE.")
        st.stop()
    if np.isinf(X_train_data).any():
        st.error("SMOTE PRE-CHECK: Inf terdeteksi di X_train_data tepat sebelum SMOTE.")
        st.stop()
    if not np.issubdtype(X_train_data.dtype, np.number):
        st.error(f"SMOTE PRE-CHECK: X_train_data bukan tipe numerik. Tipe: {X_train_data.dtype}")
        st.stop()
    if X_train_data.ndim != 2:
        st.error(f"SMOTE PRE-CHECK: X_train_data harus 2D untuk SMOTE. Dimensi: {X_train_data.ndim}")
        st.stop()
    if y_train_data.ndim != 1:
        st.error(f"SMOTE PRE-CHECK: y_train_data harus 1D untuk SMOTE. Dimensi: {y_train_data.ndim}")
        st.stop()
    
    unique_labels, counts = np.unique(y_train_data, return_counts=True)
    # st.info(f"Distribusi label di y_train_data sebelum SMOTE: {dict(zip(unique_labels, counts))}") # Dihapus
    if len(unique_labels) < 2:
        st.error("SMOTE membutuhkan setidaknya 2 kelas dalam data latih untuk oversampling. Hentikan aplikasi.")
        st.stop()
    
    min_samples_smote = 2
    for label, count in zip(unique_labels, counts):
        if count < min_samples_smote:
            st.error(f"Kelas '{label}' hanya memiliki {count} sampel. SMOTE membutuhkan minimal {min_samples_smote} sampel untuk setiap kelas.")
            st.error("Tidak dapat menerapkan SMOTE karena kelas minoritas terlalu kecil. Hentikan aplikasi.")
            st.stop()

    try:
        smote = SMOTE(random_state=42)
        X_train_smoted, y_train_smoted = smote.fit_resample(X_train_data, y_train_data)
        # st.info(f"END SMOTE: Bentuk data setelah SMOTE: X_train_smoted: {X_train_smoted.shape}, y_train_smoted: {y_train_smoted.shape}") # Dihapus
        return X_train_smoted, y_train_smoted
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menerapkan SMOTE: {e}")
        st.exception(e)
        st.stop()

X_train_smote, y_train_smote = apply_smote(X_train, y_train)


# --- START: INI BLOK KODE HASIL TUNING DARI COLAB YANG BENAR ---
gnb_best_params_from_colab = {'var_smoothing': np.float64(1.0)}
gnb_best_score_from_colab = 0.7434

best_model_gnb_from_colab = GaussianNB(**gnb_best_params_from_colab)
best_model_gnb_from_colab.fit(X_train_smote, y_train_smote)

svm_best_params_from_colab = {'C': 100, 'kernel': 'rbf'}
svm_best_score_from_colab = 0.8765

best_model_svm_from_colab = SVC(probability=True, **svm_best_params_from_colab)
best_model_svm_from_colab.fit(X_train_smote, y_train_smote)


if 'best_model_gnb' not in st.session_state:
    st.session_state['best_model_gnb'] = best_model_gnb_from_colab
    st.session_state['best_params_gnb'] = gnb_best_params_from_colab
    st.session_state['best_score_gnb'] = gnb_best_score_from_colab

if 'best_model_svm' not in st.session_state:
    st.session_state['best_model_svm'] = best_model_svm_from_colab
    st.session_state['best_params_svm'] = svm_best_params_from_colab
    st.session_state['best_score_svm'] = svm_best_score_from_colab
# --- END: INI BLOK KODE HASIL TUNING DARI COLAB ---


# Struktur Dashboard Utama
st.markdown(
    """
    <style>
    /* Mengatur warna latar belakang halaman utama Streamlit */
    .stApp {
        background-color: #0E1117;
    }

    .main-title {
        font-size: 3em;
        text-align: center;
        color: #FFFFFF;
        background-color: transparent;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
        border: none;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        margin-top: 30px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #333333;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        margin-right: 0px;
        border: none;
        cursor: pointer;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        background-color: #555555;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #1560BD;
        color: white;
    }
    .st-emotion-cache-1wv7k0o {
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 class='main-title'>Klasifikasi Unggahan Terkait Isu Korupsi dan <br> Perbandingan Algoritma <span style='color:#1560BD;'>Naive Bayes Classifier dan Support Vector Machine (SVM)</span></h1>",
    unsafe_allow_html=True
)

# Buat tab untuk berbagai bagian dashboard
tab1, tab_sentimen_tahun, tab2, tab3, tab4 = st.tabs(["Distribusi Data", "Sentimen per Tahun", "Kata Populer & Wordcloud", "Performa Model", "Evaluasi Uji Rinci"])

# --- Tab 1: Distribusi Data ---
with tab1:
    st.header("Distribusi Data: Label & Pembagian Train/Test")

    col_left_content, col_split_info = st.columns([1, 1])

    with col_left_content:
        st.subheader("Jumlah Data Keseluruhan")
        label_counts_df = pd.Series(y_labels_original).value_counts().reset_index()
        label_counts_df.columns = ['Label', 'Jumlah Data']
        label_mapping = {0: 'Negatif', 1: 'Positif'}
        label_counts_df['Label'] = label_counts_df['Label'].map(label_mapping)
        label_counts_df['No'] = range(1, len(label_counts_df) + 1)
        label_counts_df = label_counts_df[['No', 'Label', 'Jumlah Data']]

        table_html = "<table style='width:100%; text-align:center; border-collapse: collapse;'>"
        table_html += "<thead><tr>"
        for col in label_counts_df.columns:
            table_html += f"<th style='border: 1px solid #ddd; padding: 8px; background-color:#333; color:white;'>{col}</th>"
        table_html += "</tr></thead><tbody>"
        for index, row in label_counts_df.iterrows():
            table_html += "<tr>"
            for item in row:
                table_html += f"<td style='border: 1px solid #ddd; padding: 8px;'>{item}</td>"
            table_html += "</tr>"
        total_data = label_counts_df['Jumlah Data'].sum()
        table_html += "<tr>"
        table_html += "<td style='border: 1px solid #ddd; padding: 8px; font-weight:bold;'></td>"
        table_html += "<td style='border: 1px solid #ddd; padding: 8px; font-weight:bold;'>Total</td>"
        table_html += f"<td style='border: 1px solid #ddd; padding: 8px; font-weight:bold;'>{total_data}</td>"
        table_html += "</tr>"
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

        st.subheader("Diagram Pie Distribusi Label Keseluruhan") 
        fig_plotly_pie = px.pie(label_counts_df,
                                 names='Label',
                                 values='Jumlah Data',
                                 title='PIE CHART Distribusi Label Keseluruhan',
                                 color_discrete_sequence=sns.color_palette('viridis', n_colors=len(label_counts_df)).as_hex(),
                                 hole=0.3,
                                 height=450,
                                 width=450
                                 )
        fig_plotly_pie.update_traces(textinfo='percent', textfont_color='white', marker=dict(line=dict(color='#000000', width=1)))
        fig_plotly_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="white",
            title_font_color="white",
            title_font_size=18
        )
        st.plotly_chart(fig_plotly_pie, use_container_width=True)

    with col_split_info:
        st.subheader("Distribusi Split Train/Test")
        total_data_rows = len(df_data)
        train_count = len(X_train)
        test_count = len(X_test)
        train_percent = (train_count / total_data_rows) * 100
        test_percent = (test_count / total_data_rows) * 100

        split_data_rows_html = [
            ("Total Data", f"{total_data_rows}"),
            ("Data Latih (Train)", f"{train_count} ({train_percent:.0f}%)"),
            ("Data Uji (Test)", f"{test_count} ({test_percent:.0f}%)")
        ]
        split_table_html = "<table style='width:100%; text-align:center; border-collapse: collapse;'>"
        split_table_html += "<thead><tr>"
        split_table_html += f"<th style='border: 1px solid #ddd; padding: 8px; background-color:#333; color:white;'>Keterangan</th>"
        split_table_html += f"<th style='border: 1px solid #ddd; padding: 8px; background-color:#333; color:white;'>Jumlah</th>"
        split_table_html += "</tr></thead><tbody>"
        for desc, value in split_data_rows_html:
            split_table_html += "<tr>"
            split_table_html += f"<td style='border: 1px solid #ddd; padding: 8px;'>{desc}</td>"
            split_table_html += f"<td style='border: 1px solid #ddd; padding: 8px;'>{value}</td>"
            split_table_html += "</tr>"
        split_table_html += "</tbody></table>"
        st.markdown(split_table_html, unsafe_allow_html=True)
        
        st.subheader("Distribusi Label Data Uji")
        y_test_labels = pd.Series(y_test).map(label_mapping)
        fig_test_bar, ax_test_bar = plt.subplots(figsize=(6, 4))
        fig_test_bar.patch.set_facecolor('white')
        ax_test_bar.set_facecolor('white')
        sns.countplot(x=y_test_labels, ax=ax_test_bar, palette='viridis')
        ax_test_bar.set_title(f'Distribusi Label Data Uji (Total: {test_count})', color='black', fontsize=12)
        ax_test_bar.set_xlabel('')
        ax_test_bar.set_ylabel('Jumlah Data', color='black')
        ax_test_bar.tick_params(axis='x', colors='black')
        ax_test_bar.tick_params(axis='y', colors='black')
        ax_test_bar.spines['bottom'].set_color('black')
        ax_test_bar.spines['left'].set_color('black')
        ax_test_bar.spines['top'].set_visible(False)
        ax_test_bar.spines['right'].set_visible(False)
        for p in ax_test_bar.patches:
            ax_test_bar.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                 ha='center', va='center', xytext=(0, 10), textcoords='offset points', color='black')
        st.pyplot(fig_test_bar)

    st.info("Data telah berhasil dimuat dan dibagi menjadi set pelatihan dan pengujian, memastikan pengambilan sampel bertingkat untuk distribusi kelas yang seimbang.")

    # --- BAGIAN VISUALISASI PENYEIMBANGAN DATA DENGAN SMOTE ---
    st.markdown("---")
    st.header("Visualisasi Penyeimbangan Data Menggunakan SMOTE")
    st.info("SMOTE (Synthetic Minority Over-sampling Technique) digunakan untuk mengatasi ketidakseimbangan kelas pada data latih dengan menghasilkan sampel baru untuk kelas minoritas.")

    col_before_smote, col_after_smote = st.columns(2)

    with col_before_smote:
        st.subheader("Distribusi Label Data Latih Sebelum SMOTE")
        train_label_before_dist = pd.Series(y_train).value_counts().sort_index()
        train_label_before_df = pd.DataFrame({
            'Label': train_label_before_dist.index.map(label_mapping),
            'Jumlah Data': train_label_before_dist.values
        })
        
        fig_before_smote = px.bar(train_label_before_df, x='Label', y='Jumlah Data', 
                                 title='Sebelum SMOTE',
                                 color='Label',
                                 color_discrete_map={'Negatif': 'coral', 'Positif': 'skyblue'},
                                 template="plotly_dark", height=400)
        fig_before_smote.update_layout(font_color="white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig_before_smote.update_traces(texttemplate='%{y}', textposition='outside')
        st.plotly_chart(fig_before_smote, use_container_width=True)

    with col_after_smote:
        st.subheader("Distribusi Label Data Latih Sesudah SMOTE")
        train_label_after_dist = pd.Series(y_train_smote).value_counts().sort_index()
        train_label_after_df = pd.DataFrame({
            'Label': train_label_after_dist.index.map(label_mapping),
            'Jumlah Data': train_label_after_dist.values
        })

        fig_after_smote = px.bar(train_label_after_df, x='Label', y='Jumlah Data', 
                                 title='Sesudah SMOTE',
                                 color='Label',
                                 color_discrete_map={'Negatif': 'coral', 'Positif': 'skyblue'},
                                 template="plotly_dark", height=400)
        fig_after_smote.update_layout(font_color="white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig_after_smote.update_traces(texttemplate='%{y}', textposition='outside')
        st.plotly_chart(fig_after_smote, use_container_width=True)

    st.subheader("Sampel Data Hasil Pembagian")
    tab_train_sample, tab_test_sample = st.tabs(["Data Latih (Train)", "Data Uji (Test)"])

    with tab_train_sample:
        st.write("Berikut adalah seluruh data latih:")
        df_train_display_for_viz = df_train_display.copy()
        if 'teks' in df_train_display_for_viz.columns:
            df_train_display_for_viz['teks'] = df_train_display_for_viz['teks'].apply(
                lambda x: " ".join(map(str, x)) if isinstance(x, list) else str(x)
            )
        df_train_display_for_viz.reset_index(drop=True, inplace=True)
        df_train_display_for_viz.index = df_train_display_for_viz.index + 1
        st.dataframe(df_train_display_for_viz, use_container_width=True)

    with tab_test_sample:
        st.write("Berikut adalah seluruh data uji:")
        df_test_display_for_viz = df_test_display.copy()
        if 'teks' in df_test_display_for_viz.columns:
            df_test_display_for_viz['teks'] = df_test_display_for_viz['teks'].apply(
                lambda x: " ".join(map(str, x)) if isinstance(x, list) else str(x)
            )
        df_test_display_for_viz.reset_index(drop=True, inplace=True)
        df_test_display_for_viz.index = df_test_display_for_viz.index + 1
        st.dataframe(df_test_display_for_viz, use_container_width=True)


# --- START: TAB BARU "Sentimen per Tahun" ---
with tab_sentimen_tahun:
    st.header("Distribusi Data Sentimen per Tahun")

    sentiment_data_per_year = {
        'year': [2023, 2024, 2025],
        'Negatif': [178, 2823, 1550],
        'Positif': [3, 38, 17]
    }
    df_sentiment_per_year = pd.DataFrame(sentiment_data_per_year)

    df_sentiment_per_year['year'] = df_sentiment_per_year['year'].astype(str)

    st.subheader("Diagram Batang Sentimen per Tahun")

    fig_year_sentiment = px.bar(
        df_sentiment_per_year,
        x='year',
        y=['Negatif', 'Positif'],
        title='Distribusi Data Sentimen per Tahun',
        labels={'value': 'Jumlah Data', 'year': 'Tahun'},
        color_discrete_map={'Negatif': 'coral', 'Positif': 'skyblue'},
        height=500,
        template="plotly_dark"
    )
    fig_year_sentiment.update_layout(
        font_color="white",
        xaxis_title_font_color="white",
        yaxis_title_font_color="white",
        legend_title_font_color="white",
        title_font_color="white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig_year_sentiment.update_traces(texttemplate='%{y}', textposition='outside')
    fig_year_sentiment.update_yaxes(rangemode="tozero")

    st.plotly_chart(fig_year_sentiment, use_container_width=True)

    st.subheader("Tabel Distribusi Sentimen per Tahun")
    df_sentiment_per_year_display = df_sentiment_per_year.set_index('year')
    st.dataframe(df_sentiment_per_year_display, use_container_width=True)
    
    st.info("Data sentimen per tahun di atas dimasukkan secara manual sesuai permintaan Anda.")

# --- END: TAB BARU "Sentimen per Tahun" ---


# --- Tab 2: Kata Populer & Wordcloud ---
with tab2:
    st.header("Kata Populer & Wordcloud")

    st.markdown("<h3 style='text-align: left; color: white;'>Kata-kata Paling Populer</h3>", unsafe_allow_html=True)
    
    list_stopwords_indo = set(stopwords.words('indonesian'))

    def get_popular_words(text, stopwords_set, top_n=10):
        if not text:
            return pd.DataFrame(columns=['Kata', 'Jumlah'])
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower()) 
        filtered_words = [word for word in words if word not in stopwords_set and len(word) > 2]
        word_counts = Counter(filtered_words)
        return pd.DataFrame(word_counts.most_common(top_n), columns=['Kata', 'Jumlah'])

    wordclouds_dict, positive_texts_combined, negative_texts_combined = generate_sentiment_wordclouds(
        df_data, text_column='teks', sentiment_column='label'
    )

    col_pop_pos, col_pop_neg = st.columns(2)

    with col_pop_pos:
        st.markdown("<h4 style='text-align: center; color: green;'>Sentimen Positif</h4>", unsafe_allow_html=True)
        popular_pos_df = get_popular_words(positive_texts_combined, list_stopwords_indo, top_n=10)
        
        if not popular_pos_df.empty:
            fig_bar_pos, ax_bar_pos = plt.subplots(figsize=(10, 6))
            
            fig_bar_pos.patch.set_facecolor('#1E1E1E') 
            ax_bar_pos.set_facecolor('#1E1E1E') 
            
            sns.barplot(x='Jumlah', y='Kata', data=popular_pos_df.sort_values(by='Jumlah', ascending=False), 
                        palette='GnBu_d', ax=ax_bar_pos) 
            
            ax_bar_pos.tick_params(axis='x', colors='white')
            ax_bar_pos.tick_params(axis='y', colors='white')
            ax_bar_pos.set_xlabel('Jumlah', color='white', fontsize=12)
            ax_bar_pos.set_ylabel('', color='white') 
            ax_bar_pos.set_title('10 Kata Paling Sering Muncul', color='white', fontsize=14)
            
            ax_bar_pos.spines['bottom'].set_color('white')
            ax_bar_pos.spines['left'].set_color('white')
            ax_bar_pos.spines['top'].set_visible(False)
            ax_bar_pos.spines['right'].set_visible(False)
            
            for p in ax_bar_pos.patches:
                width = p.get_width()
                ax_bar_pos.text(width + 20, p.get_y() + p.get_height() / 2,
                                 f'{int(width)}',
                                 va='center', color='white')
            
            plt.tight_layout()
            st.pyplot(fig_bar_pos)
        else:
            st.info("Tidak ada kata populer untuk sentimen positif.")

    with col_pop_neg:
        st.markdown("<h4 style='text-align: center; color: red;'>Sentimen Negatif</h4>", unsafe_allow_html=True)
        popular_neg_df = get_popular_words(negative_texts_combined, list_stopwords_indo, top_n=10)
        
        if not popular_neg_df.empty:
            fig_bar_neg, ax_bar_neg = plt.subplots(figsize=(10, 6))
            
            fig_bar_neg.patch.set_facecolor('#1E1E1E')
            ax_bar_neg.set_facecolor('#1E1E1E')
            
            sns.barplot(x='Jumlah', y='Kata', data=popular_neg_df.sort_values(by='Jumlah', ascending=False), 
                        palette='RdPu_d', ax=ax_bar_neg)
            
            ax_bar_neg.tick_params(axis='x', colors='white')
            ax_bar_neg.tick_params(axis='y', colors='white')
            ax_bar_neg.set_xlabel('Jumlah', color='white', fontsize=12)
            ax_bar_neg.set_ylabel('', color='white')
            ax_bar_neg.set_title('10 Kata Paling Sering Muncul', color='white', fontsize=14)
            
            ax_bar_neg.spines['bottom'].set_color('white')
            ax_bar_neg.spines['left'].set_color('white')
            ax_bar_neg.spines['top'].set_visible(False)
            ax_bar_neg.spines['right'].set_visible(False)
            
            for p in ax_bar_neg.patches:
                width = p.get_width()
                ax_bar_neg.text(width + 20, p.get_y() + p.get_height() / 2,
                                 f'{int(width)}',
                                 va='center', color='white')
            
            plt.tight_layout()
            st.pyplot(fig_bar_neg)
        else:
            st.info("Tidak ada kata populer untuk sentimen negatif.")

    st.markdown("---")
    st.subheader("Word Clouds Berdasarkan Sentimen")

    col_wc_pos, col_wc_neg = st.columns(2)

    with col_wc_pos:
        st.markdown("<h4 style='text-align: center; color: white;'>Word Cloud Sentimen Positif</h4>", unsafe_allow_html=True)
        if 'Positif' in wordclouds_dict:
            wordcloud_positif = wordclouds_dict['Positif']
            fig_pos, ax_pos = plt.subplots(figsize=(10, 5))
            ax_pos.imshow(wordcloud_positif, interpolation='bilinear')
            ax_pos.axis('off')
            ax_pos.set_title('')
            st.pyplot(fig_pos)

            buf_pos = io.BytesIO()
            fig_pos.savefig(buf_pos, format="png", bbox_inches='tight')
            buf_pos.seek(0)
            st.download_button(
                label="Unduh Word Cloud Positif",
                data=buf_pos,
                file_name="wordcloud_positif.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.info("Tidak ada Word Cloud Positif yang dihasilkan.")

    with col_wc_neg:
        st.markdown("<h4 style='text-align: center; color: white;'>Word Cloud Sentimen Negatif</h4>", unsafe_allow_html=True)
        if 'Negatif' in wordclouds_dict:
            wordcloud_negatif = wordclouds_dict['Negatif']
            fig_neg, ax_neg = plt.subplots(figsize=(10, 5))
            ax_neg.imshow(wordcloud_negatif, interpolation='bilinear')
            ax_neg.axis('off')
            ax_neg.set_title('')
            st.pyplot(fig_neg)

            buf_neg = io.BytesIO()
            fig_neg.savefig(buf_neg, format="png", bbox_inches='tight')
            buf_neg.seek(0)
            st.download_button(
                label="Unduh Word Cloud Negatif",
                data=buf_neg,
                file_name="wordcloud_negatif.png",
                mime="image/png",
                use_container_width=True
            )
        else:
            st.info("Tidak ada Word Cloud Negatif yang dihasilkan.")

# --- Tab 3: Performa Model ---
with tab3:
    st.header("Perbandingan Performa Model pada Data Latih dan Data Uji")
    st.info("Metrik performa dihitung untuk mengevaluasi seberapa baik model belajar (data latih) dan menggeneralisasi (data uji).")

    best_model_gnb = st.session_state['best_model_gnb']
    best_model_svm = st.session_state['best_model_svm']

    col_gnb_metrics, col_svm_metrics = st.columns(2)

    with col_gnb_metrics:
        st.markdown("<h4 style='text-align: left; color: white;'>Naive Bayes Classifier</h4>", unsafe_allow_html=True) 
        
        y_pred_gnb_train = best_model_gnb.predict(X_train_smote)
        accuracy_gnb_train = accuracy_score(y_train_smote, y_pred_gnb_train)
        precision_gnb_train = precision_score(y_train_smote, y_pred_gnb_train, average='weighted', zero_division=0)
        recall_gnb_train = recall_score(y_train_smote, y_pred_gnb_train, average='weighted', zero_division=0)
        f1_gnb_train = f1_score(y_train_smote, y_pred_gnb_train, average='weighted', zero_division=0)

        y_pred_gnb_test = best_model_gnb.predict(X_test)
        accuracy_gnb_test = accuracy_score(y_test, y_pred_gnb_test)
        precision_gnb_test = precision_score(y_test, y_pred_gnb_test, average='weighted', zero_division=0)
        recall_gnb_test = recall_score(y_test, y_pred_gnb_test, average='weighted', zero_division=0)
        f1_gnb_test = f1_score(y_test, y_pred_gnb_test, average='weighted', zero_division=0)

        st.subheader("Tabel Metrik Naive Bayes Classifier")
        data_gnb_table = {
            'Metrik': ['Akurasi', 'Presisi', 'Recall', 'F1-Score'],
            'Data Latih (Train)': [accuracy_gnb_train, precision_gnb_train, recall_gnb_train, f1_gnb_train],
            'Data Uji (Test)': [accuracy_gnb_test, precision_gnb_test, recall_gnb_test, f1_gnb_test]
        }
        df_gnb_metrics_table = pd.DataFrame(data_gnb_table).set_index('Metrik')
        def highlight_vals_gnb_table(s):
            if s.name in ['Data Latih (Train)', 'Data Uji (Test)']:
                return ['background-color: #0c457d; color: white' for _ in s]
            return ['background-color: #1E1E1E; color: white']

        st.dataframe(df_gnb_metrics_table.style.format("{:.4f}").apply(highlight_vals_gnb_table, axis=0), use_container_width=True)

        st.subheader("Diagram Performa Naive Bayes Classifier")
        metrics_gnb_for_plot = pd.DataFrame({
            'Metrik': ['Akurasi', 'Presisi', 'Recall', 'F1-Score', 'Akurasi', 'Presisi', 'Recall', 'F1-Score'],
            'Tipe Data': ['Data Latih (Train)'] * 4 + ['Data Uji (Test)'] * 4,
            'Skor': [accuracy_gnb_train, precision_gnb_train, recall_gnb_train, f1_gnb_train,
                     accuracy_gnb_test, precision_gnb_test, recall_gnb_test, f1_gnb_test]
        })

        fig_gnb_metrics, ax_gnb_metrics = plt.subplots(figsize=(6, 4))
        fig_gnb_metrics.patch.set_facecolor('#1E1E1E')
        ax_gnb_metrics.set_facecolor('#1E1E1E')

        sns.barplot(x='Metrik', y='Skor', hue='Tipe Data', data=metrics_gnb_for_plot, ax=ax_gnb_metrics,
                    palette='viridis')

        ax_gnb_metrics.set_title('Performa Naive Bayes Classifier', color='white', fontsize=12)
        ax_gnb_metrics.set_xlabel('')
        ax_gnb_metrics.set_ylabel('Skor', color='white')
        ax_gnb_metrics.tick_params(axis='x', colors='white')
        ax_gnb_metrics.tick_params(axis='y', colors='white')
        ax_gnb_metrics.set_ylim(0, 1.1)

        for p in ax_gnb_metrics.patches:
            ax_gnb_metrics.annotate(f'{p.get_height():.3f}', 
                                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                                     ha='center', va='center', xytext=(0, 5),
                                     textcoords='offset points', color='white', fontsize=8)

        legend = ax_gnb_metrics.legend(title='', loc='upper right', frameon=True, fontsize=8)
        legend.get_frame().set_facecolor('#1E1E1E')
        plt.setp(legend.get_texts(), color='white')

        ax_gnb_metrics.spines['bottom'].set_color('white')
        ax_gnb_metrics.spines['left'].set_color('white')
        ax_gnb_metrics.spines['top'].set_visible(False)
        ax_gnb_metrics.spines['right'].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig_gnb_metrics)


    with col_svm_metrics:
        st.markdown("<h4 style='text-align: left; color: white;'>Support Vector Machine (SVM)</h4>", unsafe_allow_html=True)

        y_pred_svm_train = best_model_svm.predict(X_train_smote)
        accuracy_svm_train = accuracy_score(y_train_smote, y_pred_svm_train)
        precision_svm_train = precision_score(y_train_smote, y_pred_svm_train, average='weighted', zero_division=0)
        recall_svm_train = recall_score(y_train_smote, y_pred_svm_train, average='weighted', zero_division=0)
        f1_svm_train = f1_score(y_train_smote, y_pred_svm_train, average='weighted', zero_division=0)

        y_pred_svm_test = best_model_svm.predict(X_test)
        accuracy_svm_test = accuracy_score(y_test, y_pred_svm_test)
        precision_svm_test = precision_score(y_test, y_pred_svm_test, average='weighted', zero_division=0)
        recall_svm_test = recall_score(y_test, y_pred_svm_test, average='weighted', zero_division=0)
        f1_svm_test = f1_score(y_test, y_pred_svm_test, average='weighted', zero_division=0)

        st.subheader("Tabel Metrik SVM")
        data_svm_table = {
            'Metrik': ['Akurasi', 'Presisi', 'Recall', 'F1-Score'],
            'Data Latih (Train)': [accuracy_svm_train, precision_svm_train, recall_svm_train, f1_svm_train],
            'Data Uji (Test)': [accuracy_svm_test, precision_svm_test, recall_svm_test, f1_svm_test]
        }
        df_svm_metrics_table = pd.DataFrame(data_svm_table).set_index('Metrik')
        def highlight_vals_svm_table(s):
            if s.name in ['Data Latih (Train)', 'Data Uji (Test)']:
                return ['background-color: #0c457d; color: white' for _ in s]
            return ['background-color: #1E1E1E; color: white']

        st.dataframe(df_svm_metrics_table.style.format("{:.4f}").apply(highlight_vals_svm_table, axis=0), use_container_width=True)

        st.subheader("Diagram Performa SVM")
        metrics_svm_for_plot = pd.DataFrame({
            'Metrik': ['Akurasi', 'Presisi', 'Recall', 'F1-Score', 'Akurasi', 'Presisi', 'Recall', 'F1-Score'],
            'Tipe Data': ['Data Latih (Train)'] * 4 + ['Data Uji (Test)'] * 4,
            'Skor': [accuracy_svm_train, precision_svm_train, recall_svm_train, f1_svm_train,
                     accuracy_svm_test, precision_svm_test, recall_svm_test, f1_svm_test]
        })

        fig_svm_metrics, ax_svm_metrics = plt.subplots(figsize=(6, 4))
        fig_svm_metrics.patch.set_facecolor('#1E1E1E')
        ax_svm_metrics.set_facecolor('#1E1E1E')

        sns.barplot(x='Metrik', y='Skor', hue='Tipe Data', data=metrics_svm_for_plot, ax=ax_svm_metrics,
                    palette='viridis')

        ax_svm_metrics.set_title('Performa SVM', color='white', fontsize=12)
        ax_svm_metrics.set_xlabel('')
        ax_svm_metrics.set_ylabel('Skor', color='white')
        ax_svm_metrics.tick_params(axis='x', colors='white')
        ax_svm_metrics.tick_params(axis='y', colors='white')
        ax_svm_metrics.set_ylim(0, 1.1)

        for p in ax_svm_metrics.patches:
            ax_svm_metrics.annotate(f'{p.get_height():.3f}', 
                                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                                     ha='center', va='center', xytext=(0, 5), 
                                     textcoords='offset points', color='white', fontsize=8)

        legend = ax_svm_metrics.legend(title='', loc='upper right', frameon=True, fontsize=8)
        legend.get_frame().set_facecolor('#1E1E1E')
        plt.setp(legend.get_texts(), color='white')

        ax_svm_metrics.spines['bottom'].set_color('white')
        ax_svm_metrics.spines['left'].set_color('white')
        ax_svm_metrics.spines['top'].set_visible(False)
        ax_svm_metrics.spines['right'].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig_svm_metrics)

    st.success("Perbandingan performa model pada data latih dan data uji telah ditampilkan.")

    st.markdown("---")

    st.subheader("Kesimpulan Performa Model Keseluruhan")
    st.info("Berikut adalah ringkasan performa model Naive Bayes Classifier dan Support Vector Machine pada data uji.") 

    summary_data = {
        'Model': ['Naive Bayes Classifier', 'Support Vector Machine'], 
        'F1-Score CV (Train)': [st.session_state['best_score_gnb'], st.session_state['best_score_svm']],
        'Akurasi (Test)': [accuracy_gnb_test, accuracy_svm_test],
        'F1-Score (Test)': [f1_gnb_test, f1_svm_test],
        'Overfitting Indikator': [f"{abs(accuracy_gnb_train - accuracy_gnb_test):.3f}", f"{abs(accuracy_svm_train - accuracy_svm_test):.3f}"]
    }
    df_summary = pd.DataFrame(summary_data)

    best_model_name = df_summary.loc[df_summary['F1-Score (Test)'].idxmax()]['Model']

    def highlight_best_model(row):
        style = [''] * len(row)
        if row['Model'] == best_model_name:
            style = ['background-color: #1560BD; color: white'] * len(row)
        return style

    st.dataframe(df_summary.style.format({
        'F1-Score CV (Train)': "{:.4f}",
        'Akurasi (Test)': "{:.4f}",
        'F1-Score (Test)': "{:.4f}"
    }).apply(highlight_best_model, axis=1), use_container_width=True)

    st.markdown(f"**Kesimpulan:** Berdasarkan F1-Score pada data uji, model **{best_model_name}** memiliki performa terbaik.")
    # st.markdown("<p style='font-size: 0.8em; color: gray;'>*'Overfitting Indikator' menunjukkan selisih antara akurasi pada data latih dan data uji. Nilai yang lebih kecil menunjukkan generalisasi yang lebih baik.</p>", unsafe_allow_html=True) # Dihapus

# --- Tab 4: Evaluasi Uji Rinci ---
with tab4:
    st.header("Evaluasi Model pada Data Uji (Test)")
    st.markdown("Menampilkan konfigurasi model, laporan klasifikasi, matriks konfusi, dan hasil prediksi rinci.")

    col_gnb_eval, col_svm_eval = st.columns(2)

    best_model_gnb = st.session_state['best_model_gnb']
    best_model_svm = st.session_state['best_model_svm']

    y_pred_nb = best_model_gnb.predict(X_test)
    accuracy_gnb_test_tab4 = accuracy_score(y_test, y_pred_nb)
    
    y_pred_svm = best_model_svm.predict(X_test)
    accuracy_svm_test_tab4 = accuracy_score(y_test, y_pred_svm)

    with col_gnb_eval:
        st.markdown(f"<h4 style='text-align: left; color: white;'>Naive Bayes Classifier</h4>", unsafe_allow_html=True)
        st.markdown(f"<span style='font-size: 0.9em; color: limegreen;'>Akurasi: `{accuracy_gnb_test_tab4:.4f}`</span>", unsafe_allow_html=True)
        st.markdown(f"Konfigurasi: `{st.session_state['best_params_gnb']}`")

        st.write("##### Laporan Klasifikasi")
        report_dict_nb = classification_report(y_test, y_pred_nb, output_dict=True, zero_division=0)
        df_report_nb = pd.DataFrame(report_dict_nb).transpose()
        df_report_nb.index.name = 'Class/Metric'
        df_report_nb = df_report_nb.reset_index()
        df_report_nb['Class/Metric'] = df_report_nb['Class/Metric'].replace({'0': 'Negatif', '1': 'Positif'})
        df_report_nb_display = df_report_nb[~df_report_nb['Class/Metric'].isin(['accuracy', 'macro avg', 'weighted avg'])]
        st.dataframe(df_report_nb_display.style.format("{:.2f}", subset=['precision', 'recall', 'f1-score', 'support']), use_container_width=True)
        
        st.write("##### Matriks Konfusi")
        cm_nb = confusion_matrix(y_test, y_pred_nb)
        fig_nb, ax_nb = plt.subplots(figsize=(7, 5))
        fig_nb.patch.set_facecolor('#1E1E1E')
        ax_nb.set_facecolor('#1E1E1E')
        sns.heatmap(cm_nb, annot=True, fmt='d', cmap='viridis', cbar=False,
                    xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'], ax=ax_nb)
        ax_nb.set_xlabel('Prediksi', color='white')
        ax_nb.set_ylabel('Aktual', color='white')
        ax_nb.tick_params(axis='x', colors='white')
        ax_nb.tick_params(axis='y', colors='white')
        ax_nb.set_title('Matriks Konfusi', color='white')
        plt.tight_layout()
        st.pyplot(fig_nb)

    with col_svm_eval:
        st.markdown(f"<h4 style='text-align: left; color: white;'>Support Vector Machine (SVM)</h4>", unsafe_allow_html=True)
        st.markdown(f"<span style='font-size: 0.9em; color: limegreen;'>Akurasi: `{accuracy_svm_test_tab4:.4f}`</span>", unsafe_allow_html=True)
        st.markdown(f"Konfigurasi: `{st.session_state['best_params_svm']}`")

        st.write("##### Laporan Klasifikasi")
        report_dict_svm = classification_report(y_test, y_pred_svm, output_dict=True, zero_division=0)
        df_report_svm = pd.DataFrame(report_dict_svm).transpose()
        df_report_svm.index.name = 'Class/Metric'
        df_report_svm = df_report_svm.reset_index()
        df_report_svm['Class/Metric'] = df_report_svm['Class/Metric'].replace({'0': 'Negatif', '1': 'Positif'})
        df_report_svm_display = df_report_svm[~df_report_svm['Class/Metric'].isin(['accuracy', 'macro avg', 'weighted avg'])]
        st.dataframe(df_report_svm_display.style.format("{:.2f}", subset=['precision', 'recall', 'f1-score', 'support']), use_container_width=True)
        
        st.write("##### Matriks Konfusi")
        cm_svm = confusion_matrix(y_test, y_pred_svm)
        fig_svm, ax_svm = plt.subplots(figsize=(7, 5))
        fig_svm.patch.set_facecolor('#1E1E1E')
        ax_svm.set_facecolor('#1E1E1E')
        sns.heatmap(cm_svm, annot=True, fmt='d', cmap='viridis', cbar=False,
                    xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'], ax=ax_svm)
        ax_svm.set_xlabel('Prediksi', color='white')
        ax_svm.set_ylabel('Aktual', color='white')
        ax_svm.tick_params(axis='x', colors='white')
        ax_svm.tick_params(axis='y', colors='white')
        ax_svm.set_title('Matriks Konfusi', color='white')
        plt.tight_layout()
        st.pyplot(fig_svm)

    st.markdown("---")

    st.subheader("Tabel Hasil Prediksi Data Uji")
    st.info("Berikut adalah sampel data uji beserta label aktual dan prediksi dari kedua model.")

    df_results = df_test_display.copy()
    df_results['True Label'] = df_results['label'].map({0: 'Negatif', 1: 'Positif'})
    df_results['Predicted Label (Naive Bayes Classifier)'] = pd.Series(y_pred_nb).map({0: 'Negatif', 1: 'Positif'})
    df_results['Predicted Label (SVM)'] = pd.Series(y_pred_svm).map({0: 'Negatif', 1: 'Positif'})

    display_cols = ['teks', 'True Label', 'Predicted Label (Naive Bayes Classifier)', 'Predicted Label (SVM)'] 
    
    df_results_display = df_results[display_cols].copy()
    if 'teks' in df_results_display.columns:
        df_results_display['teks'] = df_results_display['teks'].apply(
            lambda x: " ".join(map(str, x)) if isinstance(x, list) else str(x)
        )

    st.dataframe(df_results_display, use_container_width=True)

    st.markdown("---")

    st.subheader("Kesimpulan Performa Model Keseluruhan pada Data Uji")
    st.info("Ringkasan metrik performa utama untuk setiap model pada data uji.")

    accuracy_gnb_test_final = accuracy_score(y_test, y_pred_nb)
    precision_gnb_test_final = precision_score(y_test, y_pred_nb, average='weighted', zero_division=0)
    recall_gnb_test_final = recall_score(y_test, y_pred_nb, average='weighted', zero_division=0)
    f1_gnb_test_final = f1_score(y_test, y_pred_nb, average='weighted', zero_division=0)

    accuracy_svm_test_final = accuracy_score(y_test, y_pred_svm)
    precision_svm_test_final = precision_score(y_test, y_pred_svm, average='weighted', zero_division=0)
    recall_svm_test_final = recall_score(y_test, y_pred_svm, average='weighted', zero_division=0)
    f1_svm_test_final = f1_score(y_test, y_pred_svm, average='weighted', zero_division=0)

    summary_data_final = {
        'Model': ['Naive Bayes Classifier', 'Support Vector Machine'], 
        'Akurasi (Test)': [accuracy_gnb_test_final, accuracy_svm_test_final],
        'Presisi (Test)': [precision_gnb_test_final, precision_svm_test_final],
        'Recall (Test)': [recall_gnb_test_final, recall_svm_test_final],
        'F1-Score (Test)': [f1_gnb_test_final, f1_svm_test_final]
    }
    df_summary_final = pd.DataFrame(summary_data_final)

    best_model_name_final = df_summary_final.loc[df_summary_final['F1-Score (Test)'].idxmax()]['Model']

    def highlight_best_model_final(row):
        style = [''] * len(row)
        if row['Model'] == best_model_name_final:
            style = ['background-color: #1560BD; color: white'] * len(row)
        return style

    st.dataframe(df_summary_final.style.format({
        'Akurasi (Test)': "{:.4f}",
        'Presisi (Test)': "{:.4f}",
        'Recall (Test)': "{:.4f}",
        'F1-Score (Test)': "{:.4f}"
    }).apply(highlight_best_model_final, axis=1), use_container_width=True)

    st.markdown(f"**Kesimpulan Akhir:** Berdasarkan metrik performa pada data uji, model **{best_model_name_final}** menunjukkan kinerja yang paling optimal.")
    
    st.success("Evaluasi model secara rinci telah selesai.")
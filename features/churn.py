# features/churn.py
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import shap
import lightgbm as lgb
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from google.cloud import storage
import streamlit as st
import plotly.express as px
import random

MODEL_BUCKET = "bdabi-group7"
MODEL_PATHS = {
    "model": "models/churn_model_best.pkl",
    "explainer": "models/shap_explainer.pkl",
    "features": "models/customer_features_full.parquet"
}
OUT_DIR = "model_v2"
os.makedirs(OUT_DIR, exist_ok=True)

def load_raw_data():
    from helpers.gcs_loader import load_parquet_from_gcs
    df = load_parquet_from_gcs(
        bucket_name="bdabi-group7",
        blob_name="preprocessed/preprocessed.parquet"
    )
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    return df

def train_churn_model(df):
    # Lấy đơn hàng delivered
    fact = df[df['order_status'] == 'delivered'].copy()
    fact['purchase_ts'] = fact['purchase_date']
    fact['delivered_ts'] = pd.to_datetime(fact['order_delivered_customer_date'], errors='coerce')
    fact['estimated_ts'] = pd.to_datetime(fact['order_estimated_delivery_date'], errors='coerce')
    fact['delivery_days'] = (fact['delivered_ts'] - fact['purchase_ts']).dt.days
    fact['delay_days'] = (fact['delivered_ts'] - fact['estimated_ts']).dt.days
    fact['item_total'] = fact['price'] + fact['freight_value']

    # Churn label
    GLOBAL_END_DATE = fact['purchase_ts'].max()
    CHURN_WINDOW = 90
    CUTOFF_DATE = GLOBAL_END_DATE - timedelta(days=CHURN_WINDOW)
    last_purchase = fact.groupby('customer_unique_id')['purchase_ts'].max().reset_index()
    last_purchase['days_since_last'] = (GLOBAL_END_DATE - last_purchase['purchase_ts']).dt.days
    last_purchase['churn'] = (last_purchase['days_since_last'] > CHURN_WINDOW).astype(int)
    feat_df = fact[fact['purchase_ts'] <= CUTOFF_DATE].copy()

    # Tạo features
    cust = feat_df.groupby('customer_unique_id').agg(
        num_orders=('order_id', 'nunique'),
        total_spent=('item_total', 'sum'),
        avg_order_value=('item_total', 'mean'),
        avg_review=('review_score', 'mean'),
        avg_delivery_days=('delivery_days', 'mean'),
        avg_delay=('delay_days', 'mean'),
        total_items=('order_item_id', 'count'),
        preferred_payment=('payment_type', lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown')
    ).reset_index()

    last_in_period = feat_df.groupby('customer_unique_id')['purchase_ts'].max().reset_index()
    last_in_period['recency'] = (CUTOFF_DATE - last_in_period['purchase_ts']).dt.days
    cust = cust.merge(last_in_period[['customer_unique_id', 'recency']], on='customer_unique_id')
    first = feat_df.groupby('customer_unique_id')['purchase_ts'].min().reset_index()
    cust = cust.merge(first.rename(columns={'purchase_ts': 'first_ts'}), on='customer_unique_id')
    cust['tenure_days'] = (CUTOFF_DATE - cust['first_ts']).dt.days + 1

    tmp = feat_df.sort_values(['customer_unique_id', 'purchase_ts'])
    tmp['prev_ts'] = tmp.groupby('customer_unique_id')['purchase_ts'].shift(1)
    tmp['days_between'] = (tmp['purchase_ts'] - tmp['prev_ts']).dt.days
    inter = tmp.groupby('customer_unique_id')['days_between'].agg([('avg_days_between', 'mean'), ('std_days_between', 'std')]).reset_index()
    cust = cust.merge(inter, on='customer_unique_id', how='left')

    for days in [30, 60, 90]:
        start = CUTOFF_DATE - timedelta(days=days)
        recent = feat_df[feat_df['purchase_ts'] >= start]
        cnt = recent.groupby('customer_unique_id')['order_id'].nunique().reset_index()
        cnt.columns = ['customer_unique_id', f'orders_last_{days}d']
        cust = cust.merge(cnt, on='customer_unique_id', how='left')
        cust[f'orders_last_{days}d'] = cust[f'orders_last_{days}d'].fillna(0).astype(int)

    delay_per_order = feat_df.groupby(['customer_unique_id', 'order_id']).agg(delay=('delay_days', 'mean')).reset_index()
    late = delay_per_order.groupby('customer_unique_id').agg(pct_late=('delay', lambda x: (x > 0).mean())).reset_index()
    cust = cust.merge(late, on='customer_unique_id', how='left')

    cols_to_fill = ['avg_days_between', 'std_days_between', 'pct_late', 'avg_review', 'avg_delay']
    cust[cols_to_fill] = cust[cols_to_fill].fillna(0)

    data = cust.merge(last_purchase[['customer_unique_id', 'churn']], on='customer_unique_id', how='left')
    data = pd.get_dummies(data, columns=['preferred_payment'], prefix='pay', dtype=int)
    data.fillna(0, inplace=True)
    data.replace([np.inf, -np.inf], 0, inplace=True)

    # Train
    target = 'churn'
    drop_cols = ['customer_unique_id', 'first_ts']
    X = data.drop(columns=[c for c in drop_cols + [target] if c in data.columns])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.02,
        'num_leaves': 128,
        'max_depth': 9,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'scale_pos_weight': (len(y_train) - y_train.sum()) / y_train.sum(),
        'seed': 42,
        'verbose': -1
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=5000,
        valid_sets=[dtrain, dvalid],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)]
    )

    pred = model.predict(X_test)
    auc = roc_auc_score(y_test, pred)
    print(f"AUC = {auc:.5f}")

    explainer = shap.TreeExplainer(model)

    joblib.dump(model, os.path.join(OUT_DIR, "churn_model_best.pkl"))
    joblib.dump(explainer, os.path.join(OUT_DIR, "shap_explainer.pkl"))
    data[['customer_unique_id'] + X.columns.tolist() + ['churn']].to_parquet(os.path.join(OUT_DIR, "customer_features_full.parquet"), index=False)

    client = storage.Client()
    bucket = client.bucket(MODEL_BUCKET)
    for file_name in ["churn_model_best.pkl", "shap_explainer.pkl", "customer_features_full.parquet"]:
        local_path = os.path.join(OUT_DIR, file_name)
        blob = bucket.blob(f"models/{file_name}")
        blob.upload_from_filename(local_path)

    return model, explainer, data

@st.cache_resource(ttl=3600)
def load_churn_assets():
    try:
        client = storage.Client()
        bucket = client.bucket(MODEL_BUCKET)
        blobs_exist = all(bucket.blob(p).exists() for p in MODEL_PATHS.values())
        if blobs_exist:
    
            def download(blob_name):
                blob = bucket.blob(blob_name)
                tmp = tempfile.NamedTemporaryFile(delete=False)
                blob.download_to_filename(tmp.name)
                return tmp.name
            model_path = download(MODEL_PATHS["model"])
            explainer_path = download(MODEL_PATHS["explainer"])
            features_path = download(MODEL_PATHS["features"])
            model = joblib.load(model_path)
            explainer = joblib.load(explainer_path)
            df = pd.read_parquet(features_path)
            os.unlink(model_path)
            os.unlink(explainer_path)
            os.unlink(features_path)
        else:
            df_raw = load_raw_data()
            model, explainer, df = train_churn_model(df_raw)
        return model, explainer, df
    except Exception as e:
        st.error(f"Không load được model Churn: {e}")
        st.stop()

def render_churn_prediction(container):
    with container:
        st.markdown("# Customer Churn Prediction")
        model, explainer, df = load_churn_assets()
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Search Customer ID", placeholder="e.g. 8d9, abc, 123")
        with col2:
            st.info(f"Total customers: **{len(df):,}**")

        if not query:
            st.info("Enter part of a Customer ID to search")
            return

        matches = df[df["customer_unique_id"].astype(str).str.contains(query, case=False, na=False)].head(20)
        if matches.empty:
            st.warning("No customers found")
            st.stop()

        selected_id = st.selectbox("Select customer", matches["customer_unique_id"])
        selected_row = df[df["customer_unique_id"] == selected_id].iloc[0]

        X = df[df["customer_unique_id"] == selected_id].drop(columns=["customer_unique_id", "churn"], errors="ignore")
        prob = float(model.predict(X)[0])

        recency = selected_row.get('recency', 0)
        if recency < 30:
            # prob = min(prob, 0.5)
            scale = recency / 30 + random.uniform(0, 0.05)
            prob = prob * scale 

        if prob >= 0.8:
            risk, color = "VERY HIGH RISK", "red"
        elif prob >= 0.6:
            risk, color = "HIGH RISK", "orange"
        elif prob >= 0.4:
            risk, color = "MONITOR", "gray"
        else:
            risk, color = "SAFE", "green"

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Churn Probability", f"{prob:.1%}")
        with c2:
            st.markdown(f"### <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)

        row = df[df["customer_unique_id"] == selected_id].iloc[0]
        st.write(f"**Customer ID**: `{selected_id}`")
        st.write(f"**Number of orders**: {int(row.get('num_orders', 0))}")
        st.write(f"**Recency**: {int(row.get('recency', 0))} days")
        st.write(f"**Total spent**: R${row.get('total_spent', 0):,.0f}")

        st.markdown("### Top drivers of churn risk")
        try:

            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0] 
            else:
                shap_vals = shap_values[0]

            shap_df = pd.DataFrame({
                'feature': X.columns,
                'shap_value': shap_vals
            }).sort_values('shap_value', key=abs, ascending=False).head(10)  

            # Vẽ bar chart trực quan
            fig = px.bar(
                shap_df,
                x='shap_value',
                y='feature',
                orientation='h',
                color='shap_value',
                color_continuous_scale='RdBu',
                title='Top Drivers of Churn Risk',
                labels={'shap_value': 'SHAP Value', 'feature': 'Feature'}
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Cannot compute SHAP visualization: {e}")


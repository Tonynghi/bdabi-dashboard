# Big Data Analysis and Business Intelligence Dashboard

This project is a simple Streamlit dashboard application for Big Data Analysis and Business Intelligence assignment of group 6. It demonstrates how to create a basic web app using Streamlit and Python. The app displays visualizes business intelligence data through an intuitive dashboard to uplift e-commerce .

## How to Run

1. Make sure you have Python installed on your system.
2. Install Streamlit if you haven't already:
   ```powershell
   pip install streamlit
   ```
3. Add required configuration files. Since we have connection to cloud storage, the key config is needed in order to make connection. Make sure you have these files in your project since they are secret files and will be ignored by Git.

```bash
bdabi-dashboar
├──.streamlit
│   └──secrets.toml
├── gcs-key.json
```

If you don't have them yet. Please contact us to get the files.

3. Run the application using the following command in your project directory:
   ```powershell
   streamlit run app.py
   ```
4. Your default web browser will open with the dashboard.

## File Structure

- `app.py`: Main Streamlit application file.
- `README.MD`: Project documentation and instructions.

## Authors

- Nghiem Pham Vy Nghi
- Pham Huynh Bao Dai
- Le Nhan Van

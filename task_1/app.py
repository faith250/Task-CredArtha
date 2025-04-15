# app.py - FastAPI implementation for Financial Data API
from fastapi import FastAPI, Query, Path, HTTPException
from typing import List, Optional
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sqlalchemy import create_engine, text
import json
import os

app = FastAPI(title="CredArtha Financial Data API", 
              description="API for financial data analysis and insights",
              version="1.0.0")

# Configuration
DB_CONNECTION = "sqlite:///finance_db.sqlite"  # Update with your actual SQL connection

# Load models
try:
    # Load transaction categorization model
    if os.path.exists("transaction_model.pkl"):
        with open("transaction_model.pkl", "rb") as f:
            vectorizer, category_model = pickle.load(f)
    else:
        print("Warning: transaction_model.pkl not found")
        vectorizer, category_model = None, None
    
    # Load risk model
    if os.path.exists("risk_model.pkl"):
        with open("risk_model.pkl", "rb") as f:
            risk_model, risk_features = pickle.load(f)
    else:
        print("Warning: risk_model.pkl not found")
        risk_model, risk_features = None, None
    
    models_loaded = (vectorizer is not None and category_model is not None and 
                    risk_model is not None and risk_features is not None)
    
    if models_loaded:
        print("Models loaded successfully")
    else:
        print("Some models failed to load")
except Exception as e:
    print(f"Error loading models: {e}")
    import traceback
    traceback.print_exc()
    models_loaded = False
    vectorizer, category_model, risk_model, risk_features = None, None, None, None

# Explanation function - same as in the notebook
def explain_risk(user_data, model=risk_model, features=risk_features):
    """Explain risk factors for a specific user"""
    if model is None or features is None:
        # Return a dummy explanation if models aren't loaded
        return {
            "risk_probability": 0.5,
            "risk_label": "Unknown",
            "top_factors": [
                {"factor": "model_unavailable", "value": 0, "impact": 1.0, "direction": "unknown"}
            ]
        }
    
    # Ensure the data has the right format
    if isinstance(user_data, dict):
        user_features = {f: user_data.get(f, 0) for f in features}
        user_df = pd.DataFrame([user_features])
    elif isinstance(user_data, pd.Series):
        user_df = pd.DataFrame([user_data[features]])
    else:
        user_df = pd.DataFrame([user_data])[features]
    
    # Get prediction
    risk_prob = model.predict_proba(user_df)[0, 1]
    risk_label = "High Risk" if risk_prob > 0.5 else "Low Risk"
    
    # Create rule-based explanation
    explanation = {
        "risk_probability": float(risk_prob),
        "risk_label": risk_label,
        "top_factors": []
    }
    
    # Rule-based factors
    user_values = user_df.iloc[0]
    factor_list = []
    
    # Credit score factor
    if user_values['credit_score'] < 650:
        factor_list.append({
            "factor": "credit_score",
            "value": float(user_values['credit_score']),
            "impact": 0.5,
            "direction": "increased"
        })
    else:
        factor_list.append({
            "factor": "credit_score",
            "value": float(user_values['credit_score']),
            "impact": 0.3,
            "direction": "decreased"
        })
        
    # Utilization factor
    if user_values['credit_utilization_pct'] > 70:
        factor_list.append({
            "factor": "credit_utilization_pct",
            "value": float(user_values['credit_utilization_pct']),
            "impact": 0.4,
            "direction": "increased"
        })
    elif user_values['credit_utilization_pct'] > 30:
        factor_list.append({
            "factor": "credit_utilization_pct",
            "value": float(user_values['credit_utilization_pct']),
            "impact": 0.2,
            "direction": "increased"
        })
    else:
        factor_list.append({
            "factor": "credit_utilization_pct",
            "value": float(user_values['credit_utilization_pct']),
            "impact": 0.2,
            "direction": "decreased"
        })
        
    # Delinquencies factor
    if user_values['delinquencies'] > 0:
        factor_list.append({
            "factor": "delinquencies",
            "value": float(user_values['delinquencies']),
            "impact": 0.6,
            "direction": "increased"
        })
    else:
        factor_list.append({
            "factor": "delinquencies",
            "value": float(user_values['delinquencies']),
            "impact": 0.1,
            "direction": "decreased"
        })
    
    # Sort factors by impact
    factor_list.sort(key=lambda x: x['impact'], reverse=True)
    explanation["top_factors"] = factor_list
    
    return explanation

# Database connection helper
def get_db_connection():
    try:
        engine = create_engine(DB_CONNECTION)
        return engine
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

# Helper functions
def get_transactions(start_date=None, end_date=None, category=None, limit=100):
    """Fetch transactions from database with filters"""
    try:
        engine = get_db_connection()
        
        if not engine:
            # Fallback to CSV if database connection fails
            if os.path.exists('cleaned_transactions.csv'):
                df = pd.read_csv('cleaned_transactions.csv')
            elif os.path.exists('financial_transactions.csv'):
                df = pd.read_csv('financial_transactions.csv')
            else:
                return pd.DataFrame()  # Empty DataFrame if no data found
            
            # Apply filters
            if start_date:
                df = df[df['date'] >= start_date]
            if end_date:
                df = df[df['date'] <= end_date]
            if category:
                if 'nlp_category' in df.columns:
                    df = df[df['nlp_category'] == category]
                elif 'true_category' in df.columns:  # Fallback to true category
                    df = df[df['true_category'] == category]
                
            return df.sort_values('date', ascending=False).head(limit)
        
        # Build SQL query
        query = "SELECT * FROM transactions WHERE 1=1"
        
        if start_date:
            query += f" AND date >= '{start_date}'"
        if end_date:
            query += f" AND date <= '{end_date}'"
        if category:
            query += f" AND nlp_category = '{category}'"
        
        query += f" ORDER BY date DESC LIMIT {limit}"
        
        return pd.read_sql(query, engine)
    
    except Exception as e:
        print(f"Error fetching transactions: {e}")
        # Last resort fallback
        return pd.DataFrame()

def get_bureau_data(user_id):
    """Fetch credit bureau data for a user"""
    try:
        engine = get_db_connection()
        
        if not engine:
            # Fallback to CSV if database connection fails
            if os.path.exists('cleaned_bureau.csv'):
                df = pd.read_csv('cleaned_bureau.csv')
            elif os.path.exists('credit_bureau_data.csv'):
                df = pd.read_csv('credit_bureau_data.csv')
            else:
                return None
                
            result = df[df['user_id'] == user_id]
            
            if result.empty:
                return None
                
            return result.iloc[0].to_dict()
        
        query = f"SELECT * FROM credit_bureau WHERE user_id = '{user_id}'"
        result = pd.read_sql(query, engine)
        
        if result.empty:
            return None
        
        return result.iloc[0].to_dict()
    
    except Exception as e:
        print(f"Error fetching bureau data: {e}")
        return None

def predict_category(description):
    """Predict transaction category using loaded model"""
    if not models_loaded or vectorizer is None or category_model is None:
        return "Uncategorized"
    
    # Preprocess text
    def preprocess_text(text):
        # Simple preprocessing (without NLTK dependency)
        words = str(text).lower().split()
        return " ".join(words)
    
    processed = preprocess_text(description)
    
    # Vectorize and predict
    vec_text = vectorizer.transform([processed])
    category = category_model.predict(vec_text)[0]
    
    return category

# API endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "api": "CredArtha Financial Data API",
        "version": "1.0.0",
        "status": "online",
        "models_loaded": models_loaded
    }

@app.get("/transactions/")
async def read_transactions(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    category: Optional[str] = Query(None, description="Transaction category"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records")
):
    """Get transactions filtered by date range and category"""
    try:
        transactions = get_transactions(start_date, end_date, category, limit)
        if transactions.empty:
            return []
        return transactions.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transactions/summary/")
async def transactions_summary(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """Get a summary of spending by category"""
    try:
        transactions = get_transactions(start_date, end_date, limit=10000)
        
        if transactions.empty:
            return {
                "spending_by_category": [],
                "income_by_category": [],
                "total_spending": 0,
                "total_income": 0,
                "net_cashflow": 0
            }
        
        # Determine which category column to use
        category_col = 'nlp_category' if 'nlp_category' in transactions.columns else 'true_category'
        
        # Get spending by category
        spending = transactions[transactions['amount'] < 0].copy()
        if not spending.empty:
            category_summary = spending.groupby(category_col)['amount'].agg(['sum', 'count']).reset_index()
            category_summary['sum'] = category_summary['sum'].abs()  # Convert to positive for easier interpretation
        else:
            category_summary = pd.DataFrame(columns=[category_col, 'sum', 'count'])
        
        # Get income summary
        income = transactions[transactions['amount'] > 0].copy()
        if not income.empty:
            income_summary = income.groupby(category_col)['amount'].agg(['sum', 'count']).reset_index()
        else:
            income_summary = pd.DataFrame(columns=[category_col, 'sum', 'count'])
        
        return {
            "spending_by_category": category_summary.to_dict('records'),
            "income_by_category": income_summary.to_dict('records'),
            "total_spending": float(spending['amount'].sum() * -1 if not spending.empty else 0),
            "total_income": float(income['amount'].sum() if not income.empty else 0),
            "net_cashflow": float(transactions['amount'].sum())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/credit/{user_id}")
async def read_credit_profile(
    user_id: str = Path(..., description="User ID to lookup")
):
    """Get credit bureau data for a specific user"""
    try:
        bureau_data = get_bureau_data(user_id)
        
        if not bureau_data:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        return bureau_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/risk/{user_id}")
async def get_risk_assessment(
    user_id: str = Path(..., description="User ID to assess")
):
    """Get risk assessment with explanations for a specific user"""
    try:
        # Get user data
        bureau_data = get_bureau_data(user_id)
        
        if not bureau_data:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        # Get risk assessment
        if not models_loaded:
            explanation = {
                "risk_probability": 0.5,
                "risk_label": "Unknown (models not loaded)",
                "top_factors": []
            }
        else:
            # Calculate risk and explanation
            explanation = explain_risk(bureau_data, risk_model, risk_features)
        
        # Return combined result
        return {
            "user_id": user_id,
            "profile": bureau_data,
            "risk_assessment": explanation
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/categorize/")
async def categorize_transaction(transaction: dict):
    """Categorize a new transaction using the trained model"""
    try:
        if not models_loaded:
            return {
                "transaction": transaction,
                "predicted_category": "Uncategorized (models not loaded)"
            }
        
        # Extract description
        if 'description' not in transaction:
            raise HTTPException(status_code=400, detail="Transaction must include a 'description' field")
        
        # Predict category
        category = predict_category(transaction['description'])
        
        # Return result
        return {
            "transaction": transaction,
            "predicted_category": category
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For direct execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
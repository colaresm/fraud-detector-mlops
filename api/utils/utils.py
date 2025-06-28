def get_params_by_prediction(data):
    monthly_income = data.get('monthly_income')
    credit_score = data.get('credit_score')
    current_debt = data.get('current_debt')
    late_payments = data.get('late_payments')
    return monthly_income,credit_score,current_debt,late_payments


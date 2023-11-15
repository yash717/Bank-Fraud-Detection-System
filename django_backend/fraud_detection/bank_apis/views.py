from django.http import JsonResponse
from .utils import load_model, predict_fraud

# Load the model when the app starts
model = load_model('./xgboost_model.pkl')

def predict_fraud_view(request):
    # Get data from the request (assuming it's sent as POST data)
    new_input_data = request.POST.dict()

    # Make prediction using the model
    prediction = predict_fraud(model, new_input_data)

    # Return the prediction as JSON response
    return JsonResponse({'fraud_prediction': prediction})

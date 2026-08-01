# ============================================================
# Project 2: AI Order Status Classification Website
# File: app.py
# ============================================================

from pathlib import Path
import json
import joblib
import pandas as pd

from flask import (
    Flask,
    render_template,
    request
)


# ============================================================
# 1. FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "model"

MODEL_FILE = MODEL_DIR / "order_status_model.pkl"

MODEL_INFORMATION_FILE = (
    MODEL_DIR / "model_information.json"
)


# ============================================================
# 3. CHECK REQUIRED FILES
# ============================================================

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        "order_status_model.pkl file nahi mili.\n"
        "Pehle train_model.py run karo."
    )

if not MODEL_INFORMATION_FILE.exists():
    raise FileNotFoundError(
        "model_information.json file nahi mili.\n"
        "Pehle train_model.py run karo."
    )


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

model = joblib.load(MODEL_FILE)


# ============================================================
# 5. LOAD MODEL INFORMATION
# ============================================================

with open(
    MODEL_INFORMATION_FILE,
    "r",
    encoding="utf-8"
) as file:

    model_information = json.load(file)


# Website dropdown options

PRODUCT_OPTIONS = model_information[
    "product_options"
]

PAYMENT_METHOD_OPTIONS = model_information[
    "payment_method_options"
]

COUPON_CODE_OPTIONS = model_information[
    "coupon_code_options"
]

REFERRAL_SOURCE_OPTIONS = model_information[
    "referral_source_options"
]

MODEL_NAME = model_information[
    "model_name"
]

CLASS_NAMES = model_information[
    "classes"
]


# ============================================================
# 6. HOME PAGE ROUTE
# ============================================================

@app.route("/")
def home():
    """
    Website ka main page show karega.
    """

    return render_template(
        "index.html",

        product_options=PRODUCT_OPTIONS,

        payment_method_options=(
            PAYMENT_METHOD_OPTIONS
        ),

        coupon_code_options=(
            COUPON_CODE_OPTIONS
        ),

        referral_source_options=(
            REFERRAL_SOURCE_OPTIONS
        ),

        model_name=MODEL_NAME
    )


# ============================================================
# 7. PREDICTION ROUTE
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():
    """
    Form se order details receive karega,
    AI model se prediction karwayega
    aur result page show karega.
    """

    try:

        # ----------------------------------------------------
        # Form values receive karna
        # ----------------------------------------------------

        product = request.form.get(
            "Product",
            ""
        ).strip()

        quantity = request.form.get(
            "Quantity",
            ""
        ).strip()

        unit_price = request.form.get(
            "UnitPrice",
            ""
        ).strip()

        payment_method = request.form.get(
            "PaymentMethod",
            ""
        ).strip()

        items_in_cart = request.form.get(
            "ItemsInCart",
            ""
        ).strip()

        coupon_code = request.form.get(
            "CouponCode",
            "No Coupon"
        ).strip()

        referral_source = request.form.get(
            "ReferralSource",
            ""
        ).strip()


        # ----------------------------------------------------
        # Required fields validation
        # ----------------------------------------------------

        if not product:
            raise ValueError(
                "Please select a product."
            )

        if not quantity:
            raise ValueError(
                "Please enter quantity."
            )

        if not unit_price:
            raise ValueError(
                "Please enter unit price."
            )

        if not payment_method:
            raise ValueError(
                "Please select payment method."
            )

        if not items_in_cart:
            raise ValueError(
                "Please enter items in cart."
            )

        if not referral_source:
            raise ValueError(
                "Please select referral source."
            )


        # ----------------------------------------------------
        # Numerical values conversion
        # ----------------------------------------------------

        quantity = int(quantity)

        unit_price = float(unit_price)

        items_in_cart = int(items_in_cart)


        # ----------------------------------------------------
        # Numerical validation
        # ----------------------------------------------------

        if quantity <= 0:
            raise ValueError(
                "Quantity zero se greater honi chahiye."
            )

        if unit_price <= 0:
            raise ValueError(
                "Unit price zero se greater honi chahiye."
            )

        if items_in_cart <= 0:
            raise ValueError(
                "Items in cart zero se greater hone chahiye."
            )


        # ----------------------------------------------------
        # Total Price calculate karna
        # ----------------------------------------------------

        total_price = round(
            quantity * unit_price,
            2
        )


        # ----------------------------------------------------
        # Model input DataFrame
        # Exact feature names training code ke mutabiq hain
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [
                {
                    "Product": product,

                    "Quantity": quantity,

                    "UnitPrice": unit_price,

                    "PaymentMethod": payment_method,

                    "ItemsInCart": items_in_cart,

                    "CouponCode": coupon_code,

                    "ReferralSource": referral_source,

                    "TotalPrice": total_price
                }
            ]
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predicted_status = model.predict(
            input_data
        )[0]


        # ----------------------------------------------------
        # Prediction confidence
        # ----------------------------------------------------

        confidence = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            confidence = round(
                max(probabilities) * 100,
                2
            )


        # ----------------------------------------------------
        # Result page
        # ----------------------------------------------------

        return render_template(
            "result.html",

            predicted_status=predicted_status,

            confidence=confidence,

            model_name=MODEL_NAME,

            product=product,

            quantity=quantity,

            unit_price=unit_price,

            payment_method=payment_method,

            items_in_cart=items_in_cart,

            coupon_code=coupon_code,

            referral_source=referral_source,

            total_price=total_price
        )


    except ValueError as error:

        return render_template(
            "index.html",

            product_options=PRODUCT_OPTIONS,

            payment_method_options=(
                PAYMENT_METHOD_OPTIONS
            ),

            coupon_code_options=(
                COUPON_CODE_OPTIONS
            ),

            referral_source_options=(
                REFERRAL_SOURCE_OPTIONS
            ),

            model_name=MODEL_NAME,

            error_message=str(error)
        )


    except Exception as error:

        print(
            f"Prediction Error: {error}"
        )

        return render_template(
            "index.html",

            product_options=PRODUCT_OPTIONS,

            payment_method_options=(
                PAYMENT_METHOD_OPTIONS
            ),

            coupon_code_options=(
                COUPON_CODE_OPTIONS
            ),

            referral_source_options=(
                REFERRAL_SOURCE_OPTIONS
            ),

            model_name=MODEL_NAME,

            error_message=(
                "Prediction ke waqt error aya. "
                "Please entered values check karo."
            )
        )


# ============================================================
# 8. APPLICATION RUN
# ============================================================

if __name__ == "__main__":

    print("\n==============================================")
    print("AI Order Status Prediction Website")
    print("==============================================")
    print(f"Loaded Model: {MODEL_NAME}")
    print("Website URL: http://127.0.0.1:5000")
    print("==============================================\n")

    app.run(
        debug=True
    )
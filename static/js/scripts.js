// ============================================================
// Project 2: AI Order Status Prediction Website
// File: static/js/script.js
// ============================================================


// ============================================================
// 1. REQUIRED HTML ELEMENTS
// ============================================================

const predictionForm = document.getElementById("predictionForm");

const quantityInput = document.getElementById("Quantity");

const unitPriceInput = document.getElementById("UnitPrice");

const totalPriceDisplay = document.getElementById(
    "TotalPriceDisplay"
);

const predictButton = document.getElementById(
    "predictButton"
);


// ============================================================
// 2. TOTAL PRICE CALCULATION
// ============================================================

function calculateTotalPrice() {

    const quantity = parseFloat(
        quantityInput.value
    ) || 0;

    const unitPrice = parseFloat(
        unitPriceInput.value
    ) || 0;

    const totalPrice = quantity * unitPrice;

    totalPriceDisplay.textContent = totalPrice.toFixed(2);
}


// Quantity change hone par total update hoga

quantityInput.addEventListener(
    "input",
    calculateTotalPrice
);


// Unit price change hone par total update hoga

unitPriceInput.addEventListener(
    "input",
    calculateTotalPrice
);


// ============================================================
// 3. BASIC FORM VALIDATION
// ============================================================

function validatePositiveNumber(inputElement, fieldName) {

    const value = Number(inputElement.value);

    if (!inputElement.value) {

        inputElement.focus();

        alert(
            `Please enter ${fieldName}.`
        );

        return false;
    }

    if (value <= 0) {

        inputElement.focus();

        alert(
            `${fieldName} must be greater than zero.`
        );

        return false;
    }

    return true;
}


// ============================================================
// 4. FORM SUBMISSION
// ============================================================

predictionForm.addEventListener(
    "submit",
    function (event) {

        const quantityIsValid = validatePositiveNumber(
            quantityInput,
            "quantity"
        );

        if (!quantityIsValid) {

            event.preventDefault();

            return;
        }

        const unitPriceIsValid = validatePositiveNumber(
            unitPriceInput,
            "unit price"
        );

        if (!unitPriceIsValid) {

            event.preventDefault();

            return;
        }

        const itemsInCartInput = document.getElementById(
            "ItemsInCart"
        );

        const itemsInCartIsValid = validatePositiveNumber(
            itemsInCartInput,
            "items in cart"
        );

        if (!itemsInCartIsValid) {

            event.preventDefault();

            return;
        }


        // Button loader show hoga

        predictButton.classList.add(
            "loading"
        );

        predictButton.disabled = true;
    }
);


// ============================================================
// 5. INITIAL TOTAL PRICE
// ============================================================

calculateTotalPrice();
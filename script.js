document.addEventListener('DOMContentLoaded', () => {
    let selectedModel = ''; // Variable to store the selected model's image source
    let selectedShirt = ''; // Variable to store the selected shirt's image source
    let selectedPant = ''; // Variable to store the selected pant's image source
    let selectedAccessory = ''; // Variable to store the selected accessory's image source
    let selectedShoes = ''; // Variable to store the selected shoes' image source

    // Elements to display the selected items
    const modelDisplay = document.getElementById('selected-model');
    const shirtDisplay = document.getElementById('selected-shirt');
    const pantDisplay = document.getElementById('selected-pant');
    const accessoryDisplay = document.getElementById('selected-accessory');
    const shoesDisplay = document.getElementById('selected-shoes');

    // Event listener for selecting a model
    document.querySelectorAll('.model').forEach(model => {
        model.addEventListener('click', () => {
            selectedModel = model.src; // Get the selected model's image source
            if (selectedModel.includes('model2.png')) {
                window.location.href = 'model_men.html'; // Redirect to model_men.html if model2.png is selected
            } else if (selectedModel.includes('model1.png')) {
                window.location.href = 'model_women.html'; // Redirect to model_women.html if model1.png is selected
            } else {
                modelDisplay.src = selectedModel; // Display the selected model's image
                resetSelections(); // Reset selections for other items
            }
        });
    });

    // Event listener for selecting an item (shirt, pant, accessory, shoes)
    document.querySelectorAll('.item').forEach(item => {
        item.addEventListener('click', () => {
            const category = item.id.split('')[0]; // Get the first letter of the item's id to determine its category
            if (category === 's' && item.id.startsWith('shirt')) {
                selectedShirt = item.src; // Get the selected shirt's image source
                shirtDisplay.src = selectedShirt; // Display the selected shirt's image
            } else if (category === 'p' && item.id.startsWith('pant')) {
                selectedPant = item.src; // Get the selected pant's image source
                pantDisplay.src = selectedPant; // Display the selected pant's image
            } else if (category === 'a' && item.id.startsWith('accessory')) {
                selectedAccessory = item.src; // Get the selected accessory's image source
                accessoryDisplay.src = selectedAccessory; // Display the selected accessory's image
            } else if (category === 's' && item.id.startsWith('shoes')) {
                selectedShoes = item.src; // Get the selected shoes' image source
                shoesDisplay.src = selectedShoes; // Display the selected shoes' image
            }
        });
    });

    // Event listener for the submit button
    document.getElementById('submit').addEventListener('click', () => {
        alert('Your style resonates beautifully with current trends. Share it in the community for feedback and let the accolades roll in!');
        window.location.href = 'model_women.html'; // Redirect to model_women.html after submission
    });

    // Function to reset selections for all items
    function resetSelections() {
        selectedShirt = ''; // Reset selected shirt
        selectedPant = ''; // Reset selected pant
        selectedAccessory = ''; // Reset selected accessory
        selectedShoes = ''; // Reset selected shoes
        shirtDisplay.src = ''; // Clear shirt display
        pantDisplay.src = ''; // Clear pant display
        accessoryDisplay.src = ''; // Clear accessory display
        shoesDisplay.src = ''; // Clear shoes display
    }
});

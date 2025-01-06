window.onload = () => {
    injectModal(); // Inject modal on page load
    checkCookie(); // Check if the cookie is set and update UI accordingly
};

const injectModal = () => {
    // Check if a <style> tag already exists
    let style = document.querySelector('style');

    if (!style) {
        // If no <style> tag exists, create one
        style = document.createElement('style');
        document.head.appendChild(style);
    }

    // Add the modal styles to the existing <style> tag or the new one
    style.textContent += `
        .content-locker-modal {
            display: block;
            position: fixed;
            z-index: 999;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.7);
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .content-locker-modal-content {
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            text-align: center;
        }
        .content-locker-blurred {
            filter: blur(8px);
            pointer-events: none;
        }
        .content-locker-hidden {
            display: none;
        }
    `;

    // Create the modal HTML structure
    const modalHTML = `
        <div id="content-locker-modal" class="content-locker-modal">
            <div class="content-locker-modal-content">
                <h2>Enter the code to access the course content</h2>
                <input type="text" id="unlock-code" placeholder="Enter unlock code">
                <button onclick="checkUnlockCode()">Unlock</button>
            </div>
            </div>
        </div>
    `;

    // Append the modal structure just below the body tag
    document.body.insertAdjacentHTML('afterbegin', modalHTML);
    console.log("Modal injected into the DOM");
};

// Function to add blurred class to all elements except the modal itself
const addBlurredContentClass = () => {
    Array.from(document.body.children).forEach((child) => {
        if (child.id !== 'content-locker-modal') {
            child.classList.add('content-locker-blurred');
        }
    });
};

const checkUnlockCode = () => {
    const enteredCode = document.getElementById("unlock-code").value;
    if (enteredCode === "unlockme") {
        document.getElementById("content-locker-modal").classList.add("content-locker-hidden"); // Hide modal
        Array.from(document.body.children).forEach((child) => {
            child.classList.remove('content-locker-blurred'); // Remove blur from all elements
        });
        document.cookie = "unlocked=true; path=/; max-age=" + (60 * 60 * 24 * 30); // Set cookie for 30 days
    } else {
        alert("Incorrect code. Please try again.");
    }
};

const checkCookie = () => {
    const cookieValue = document.cookie.split("; ").find(row => row.startsWith("unlocked="));
    if (cookieValue && cookieValue.split("=")[1] === "true") {
        document.getElementById("content-locker-modal").classList.add("content-locker-hidden"); // Hide modal if cookie exists
        Array.from(document.body.children).forEach((child) => {
            child.classList.remove('content-locker-blurred'); // Remove blur from all elements
        });
    } else {
        addBlurredContentClass(); // Blur content
    }
};
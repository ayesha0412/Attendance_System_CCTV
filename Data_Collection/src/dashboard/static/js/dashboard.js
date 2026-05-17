/**
 * Dashboard Client — SocketIO + live stats
 *
 * Connects to the Flask-SocketIO server and handles:
 * - New face capture events (adds thumbnails to gallery)
 * - Stats updates (faces count, FPS, camera status)
 * - Connection state management
 */

const MAX_GALLERY_ITEMS = 50;

// ---- SocketIO Connection ----
const socket = io();

socket.on("connect", () => {
    console.log("Dashboard connected");
    setConnectionStatus(true);
    hideVideoOverlay();
});

socket.on("disconnect", () => {
    console.log("Dashboard disconnected");
    setConnectionStatus(false);
});

// ---- New Face Event ----
socket.on("new_face", (data) => {
    addFaceToGallery(data.image, data.track_id, data.timestamp);
});

// ---- Stats Update Event ----
socket.on("stats_update", (stats) => {
    updateStats(stats);
});

// ---- UI Functions ----

function setConnectionStatus(connected) {
    const dot = document.getElementById("connection-indicator");
    const text = document.getElementById("connection-text");

    if (connected) {
        dot.className = "status-dot connected";
        text.textContent = "Connected";
    } else {
        dot.className = "status-dot disconnected";
        text.textContent = "Disconnected";
    }
}

function hideVideoOverlay() {
    const overlay = document.getElementById("video-overlay");
    if (overlay) {
        overlay.classList.add("hidden");
    }
}

function addFaceToGallery(base64Image, trackId, timestamp) {
    const gallery = document.getElementById("face-gallery");

    // Remove placeholder if present
    const placeholder = gallery.querySelector(".gallery-placeholder");
    if (placeholder) {
        placeholder.remove();
    }

    // Remove "latest" class from previous cards
    gallery.querySelectorAll(".face-card.latest").forEach((card) => {
        card.classList.remove("latest");
    });

    // Create face card
    const card = document.createElement("div");
    card.className = "face-card latest";
    card.innerHTML = `
        <img src="data:image/jpeg;base64,${base64Image}" alt="Face #${trackId}">
        <div class="face-card-info">
            <span class="track-id">#${trackId}</span>
            <span>${timestamp}</span>
        </div>
    `;

    // Insert at the top of the gallery
    gallery.insertBefore(card, gallery.firstChild);

    // Trim gallery to max size
    const cards = gallery.querySelectorAll(".face-card");
    if (cards.length > MAX_GALLERY_ITEMS) {
        cards[cards.length - 1].remove();
    }

    // Update gallery count badge
    updateGalleryCount();
}

function updateStats(stats) {
    document.getElementById("stat-today").textContent = stats.faces_today.toLocaleString();
    document.getElementById("stat-total").textContent = stats.faces_total.toLocaleString();
    document.getElementById("stat-tracks").textContent = stats.active_tracks;
    document.getElementById("stat-blur").textContent = stats.skipped_blur;
    document.getElementById("stat-fps").textContent = stats.fps;
    document.getElementById("fps-badge").textContent = `${stats.fps} FPS`;

    // Camera status
    const camEl = document.getElementById("stat-camera");
    if (stats.camera_connected) {
        camEl.textContent = "Connected";
        camEl.className = "stat-value status-connected";
    } else {
        camEl.textContent = "Disconnected";
        camEl.className = "stat-value status-disconnected";
    }
}

function updateGalleryCount() {
    const gallery = document.getElementById("face-gallery");
    const count = gallery.querySelectorAll(".face-card").length;
    document.getElementById("gallery-count").textContent = count;
}

// ---- Periodic Stats Poll (fallback if SocketIO misses) ----
setInterval(async () => {
    try {
        const resp = await fetch("/api/stats");
        if (resp.ok) {
            const stats = await resp.json();
            updateStats(stats);
        }
    } catch (e) {
        // Silently ignore fetch errors
    }
}, 5000);

// ---- Video Feed Error Handling ----
const videoFeed = document.getElementById("video-feed");
if (videoFeed) {
    videoFeed.addEventListener("load", hideVideoOverlay);
    videoFeed.addEventListener("error", () => {
        const overlay = document.getElementById("video-overlay");
        if (overlay) {
            overlay.classList.remove("hidden");
            overlay.querySelector("span").textContent = "⚠️ Camera feed unavailable";
        }
    });
}

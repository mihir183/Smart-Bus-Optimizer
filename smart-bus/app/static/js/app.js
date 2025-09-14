// Smart Bus System - Main JavaScript Application

class SmartBusApp {
    constructor() {
        this.socket = null;
        this.currentRoute = null;
        this.map = null;
        this.markers = {};
        this.realTimeData = {};
        this.updateInterval = null;

        this.init();
    }

    init() {
        this.initializeSocket();
        this.initializeEventListeners();
        this.loadInitialData();
        this.startRealTimeUpdates();
    }

    initializeSocket() {
        this.socket = io();

        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateConnectionStatus(true);
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateConnectionStatus(false);
        });

        this.socket.on('gps_update', data => this.handleGPSUpdate(data));
        this.socket.on('occupancy_update', data => this.handleOccupancyUpdate(data));
        this.socket.on('delay_update', data => this.handleDelayUpdate(data));
        this.socket.on('schedule_update', data => this.handleScheduleUpdate(data));
    }

    initializeEventListeners() {
        document.getElementById('routeSelect')?.addEventListener('change', e => {
            this.selectRoute(e.target.value);
        });

        document.querySelectorAll('.btn-refresh').forEach(btn =>
            btn.addEventListener('click', () => this.refreshData())
        );

        document.getElementById('generatePredictions')?.addEventListener('click', () => this.generatePredictions());
        document.getElementById('detectBunching')?.addEventListener('click', () => this.detectBunching());

        document.getElementById('autoRefresh')?.addEventListener('change', e => {
            this.toggleAutoRefresh(e.target.checked);
        });
    }

    async loadInitialData() {
        try {
            await Promise.all([
                this.loadRoutes(),
                this.loadBuses(),
                this.loadTrips(),
                this.loadPredictions()
            ]);
            this.updateDashboard();
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.showAlert('Error loading data', 'danger');
        }
    }

    async fetchJSON(url, options = {}) {
        const response = await fetch(url, options);
        const data = await response.json();
        if (!data.success) throw new Error(data.error || 'Unknown API error');
        return data.data;
    }

    async loadRoutes() {
        try {
            const data = await this.fetchJSON('/routes');
            this.routes = data;
            this.populateRouteSelect();
        } catch (error) {
            console.error('Error loading routes:', error);
            throw error;
        }
    }

    async loadBuses() {
        try {
            const data = await this.fetchJSON('/buses');
            this.buses = data;
            this.updateBusStatus();
        } catch (error) {
            console.error('Error loading buses:', error);
            throw error;
        }
    }

    async loadTrips(routeId = null) {
        try {
            const url = routeId ? `/api/trips?route_id=${routeId}` : '/api/trips';
            const data = await this.fetchJSON(url);
            this.trips = data;
            this.updateTripsTable();
        } catch (error) {
            console.error('Error loading trips:', error);
            throw error;
        }
    }

    async loadPredictions(routeId = null) {
        try {
            const url = routeId ? `/api/predictions?route_id=${routeId}` : '/api/predictions';
            const data = await this.fetchJSON(url);
            this.predictions = data;
            this.updatePredictionsTable();
        } catch (error) {
            console.error('Error loading predictions:', error);
            throw error;
        }
    }

    populateRouteSelect() {
        const routeSelect = document.getElementById('routeSelect');
        if (!routeSelect || !this.routes) return;

        routeSelect.innerHTML = '<option value="">Select a route...</option>';
        this.routes.forEach(route => {
            const option = document.createElement('option');
            option.value = route.id;
            option.textContent = `${route.route_number} - ${route.name}`;
            routeSelect.appendChild(option);
        });
    }

    selectRoute(routeId) {
        if (!routeId) {
            this.currentRoute = null;
            this.clearMap();
            return;
        }

        this.currentRoute = this.routes.find(r => r.id == routeId);
        this.loadRouteDetails(routeId);
        this.loadTrips(routeId);
        this.loadPredictions(routeId);
        this.initializeMap();
    }

    async loadRouteDetails(routeId) {
        try {
            const data = await this.fetchJSON(`/api/routes/${routeId}`);
            this.routeDetails = data;
            this.updateRouteInfo();
            this.updateMapWithStops();
        } catch (error) {
            console.error('Error loading route details:', error);
            this.showAlert('Error loading route details', 'danger');
        }
    }

    updateRouteInfo() {
        if (!this.routeDetails) return;
        const routeInfo = document.getElementById('routeInfo');
        if (!routeInfo) return;

        const route = this.routeDetails.route;
        routeInfo.innerHTML = `
            <h3>${route.route_number} - ${route.name}</h3>
            <p><strong>From:</strong> ${route.start_stop}</p>
            <p><strong>To:</strong> ${route.end_stop}</p>
            <p><strong>Distance:</strong> ${route.total_distance} km</p>
            <p><strong>Duration:</strong> ${route.estimated_duration} minutes</p>
        `;
    }

    updateMapWithStops() {
        if (!this.map || !this.routeDetails?.stops) return;

        Object.values(this.markers).forEach(marker => marker.remove());
        this.markers = {};

        this.routeDetails.stops.forEach((stop, i) => {
            const marker = L.marker([stop.latitude, stop.longitude])
                .addTo(this.map)
                .bindPopup(`<strong>Stop ${i + 1}</strong><br>${stop.name}<br>Code: ${stop.code || 'N/A'}`);
            this.markers[`stop_${stop.id}`] = marker;
        });

        if (this.routeDetails.stops.length > 0) {
            const group = new L.featureGroup(Object.values(this.markers));
            this.map.fitBounds(group.getBounds().pad(0.1));
        }
    }

    initializeMap() {
        if (this.map) this.map.remove();
        this.map = L.map('map').setView([40.7128, -74.0060], 13);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(this.map);
        this.updateMapWithStops();
    }

    clearMap() {
        if (this.map) {
            Object.values(this.markers).forEach(m => m.remove());
            this.markers = {};
        }
    }

    // -------------------- existing updateX, handleX, markerX, etc. remain the same --------------------

    updateConnectionStatus(connected) {
        const el = document.getElementById('connectionStatus');
        if (!el) return;
        el.innerHTML = connected
            ? '<span class="real-time-indicator">Connected</span>'
            : '<span class="alert alert-danger">Disconnected</span>';
    }

    showAlert(message, type = 'info') {
        const alertContainer = document.getElementById('alertContainer');
        if (!alertContainer) return;

        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible`;
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" onclick="this.parentElement.remove()"></button>
        `;

        alertContainer.appendChild(alert);
        setTimeout(() => alert.remove(), 5000);
    }
}

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    window.app = new SmartBusApp();
});

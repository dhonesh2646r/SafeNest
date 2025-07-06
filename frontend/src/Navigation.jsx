import React, { useState, useEffect, useRef } from 'react';
import 'leaflet/dist/leaflet.css';
import { MapContainer, TileLayer, Polyline, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';

// Fix for default marker icons in Leaflet
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
});

const API_BASE = 'http://127.0.0.1:8000';

// Updated defaults for Hyderabad (Adibatla area)
const defaultCoords = {
  current_lat: 17.2160,  // Adibatla latitude
  current_lon: 78.6139,  // Adibatla longitude
  dest_lat: 17.18985,    // Default to police station
  dest_lon: 78.64838,
};

const optimizationOptions = [
  { value: 'shortest', label: 'Shortest Path' },
  { value: 'fastest', label: 'Fastest Route' },
  { value: 'safest', label: 'Safest Path' },
  { value: 'quietest', label: 'Quietest Route' },
];

const incidentTypes = [
  { value: 'harassment', label: 'Harassment' },
  { value: 'theft', label: 'Theft Attempt' },
  { value: 'assault', label: 'Assault' },
  { value: 'suspicious', label: 'Suspicious Activity' },
  { value: 'other', label: 'Other Safety Concern' },
];

const severityLevels = [
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' },
  { value: 'emergency', label: 'Emergency' },
];

function MapDisplay({ route, currentPosition, destination }) {
  const mapRef = useRef(null);

  useEffect(() => {
    if (mapRef.current && route?.length > 0) {
      const bounds = L.latLngBounds(route);
      mapRef.current.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [route]);

  if (!route || route.length === 0) {
    return (
      <div style={{ 
        height: '400px', 
        width: '100%', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        backgroundColor: '#f0f0f0'
      }}>
        {route ? 'No route data available' : 'Loading map...'}
      </div>
    );
  }

  return (
    <MapContainer 
      ref={mapRef}
      center={route[Math.floor(route.length / 2)]} 
      zoom={14} 
      style={{ height: '400px', width: '100%' }}
      key={`map-${route.length}-${JSON.stringify(route[0])}`}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />
      <Polyline 
        positions={route}
        color="blue"
        weight={5}
        opacity={0.7}
      />
      {currentPosition && (
        <Marker position={currentPosition}>
          <Popup>Current Position</Popup>
        </Marker>
      )}
      {destination && (
        <Marker position={destination}>
          <Popup>Destination</Popup>
        </Marker>
      )}
    </MapContainer>
  );
}

function Navigation() {
  const [coords, setCoords] = useState(defaultCoords);
  const [optimizeFor, setOptimizeFor] = useState('safest');
  const [sessionId, setSessionId] = useState(null);
  const [routeData, setRouteData] = useState(null);
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [health, setHealth] = useState(null);
  const [nextCheckIn, setNextCheckIn] = useState(null);
  const [currentPosition, setCurrentPosition] = useState(null);
  const [destination, setDestination] = useState(null);
  const [showReportForm, setShowReportForm] = useState(false);
  const [incidentData, setIncidentData] = useState({
    incident_type: 'harassment',
    severity: 'medium',
    description: '',
    user_contact: '',
    location_lat: defaultCoords.current_lat,
    location_lon: defaultCoords.current_lon
  });

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (!res.ok) throw new Error('Backend unavailable');
        setHealth(await res.json());
      } catch (e) {
        setHealth(null);
        setError(e.message);
      }
    };
    
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const startNavigation = async () => {
    setLoading(true);
    setError('');
    setMessage('');
    setRouteData(null);
    
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);
      
      const res = await fetch(`${API_BASE}/start_navigation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          ...coords, 
          optimize_for: optimizeFor, 
          session_id: sessionId 
        }),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Failed to start navigation');
      }
      
      const data = await res.json();
      
      setSessionId(data.session_id);
      setRouteData({
        route: data.route,
        waypoints: data.waypoints,
        distance: data.distance,
        estimated_time: data.estimated_time
      });
      setCurrentPosition([coords.current_lat, coords.current_lon]);
      setDestination([coords.dest_lat, coords.dest_lon]);
      setMessage(data.message || 'Navigation started successfully');
      setNextCheckIn(data.next_check_in);
    } catch (e) {
      console.error('Navigation error:', e);
      setError(e.name === 'AbortError' ? 'Request timed out' : e.message);
    } finally {
      setLoading(false);
    }
  };

  const updatePosition = async () => {
    if (!sessionId) return;
    
    setLoading(true);
    setError('');
    setMessage('');
    
    try {
      const res = await fetch(`${API_BASE}/update_position`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          current_lat: coords.current_lat,
          current_lon: coords.current_lon,
          session_id: sessionId,
        }),
      });
      
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Failed to update position');
      }
      
      const data = await res.json();
      setRouteData({
        route: data.route,
        waypoints: data.waypoints,
        distance: data.distance,
        estimated_time: data.estimated_time
      });
      setCurrentPosition([coords.current_lat, coords.current_lon]);
      setMessage(data.message || 'Position updated successfully');
      setNextCheckIn(data.next_check_in);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const changeDestination = async () => {
    if (!sessionId) return;
    
    setLoading(true);
    setError('');
    setMessage('');
    
    try {
      const res = await fetch(`${API_BASE}/change_destination`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          new_dest_lat: coords.dest_lat,
          new_dest_lon: coords.dest_lon,
          session_id: sessionId,
        }),
      });
      
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Failed to change destination');
      }
      
      const data = await res.json();
      setRouteData({
        route: data.route,
        waypoints: data.waypoints,
        distance: data.distance,
        estimated_time: data.estimated_time
      });
      setDestination([coords.dest_lat, coords.dest_lon]);
      setMessage(data.message || 'Destination changed successfully');
      setNextCheckIn(data.next_check_in);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const endNavigation = async () => {
    if (!sessionId) return;
    
    setLoading(true);
    setError('');
    setMessage('');
    
    try {
      const res = await fetch(`${API_BASE}/end_navigation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });
      
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Failed to end navigation');
      }
      
      setSessionId(null);
      setRouteData(null);
      setCurrentPosition(null);
      setDestination(null);
      setMessage('Navigation ended successfully');
      setNextCheckIn(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleInput = e => {
    const { name, value } = e.target;
    setCoords(prev => ({ ...prev, [name]: parseFloat(value) || 0 }));
  };

  const handleIncidentInput = e => {
    const { name, value } = e.target;
    setIncidentData(prev => ({ ...prev, [name]: value }));
  };

  const submitIncidentReport = async () => {
    if (!sessionId) {
      setError('Please start navigation before reporting an incident');
      return;
    }
    
    setLoading(true);
    setError('');
    setMessage('');
    
    try {
      const report = {
        ...incidentData,
        session_id: sessionId,
        location_lat: coords.current_lat,
        location_lon: coords.current_lon
      };
      
      const res = await fetch(`${API_BASE}/report_incident`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(report),
      });
      
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Failed to report incident');
      }
      
      const data = await res.json();
      setMessage(data.message || 'Incident reported successfully');
      setShowReportForm(false);
      setIncidentData({
        incident_type: 'harassment',
        severity: 'medium',
        description: '',
        user_contact: '',
        location_lat: coords.current_lat,
        location_lon: coords.current_lon
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ 
      maxWidth: 800, 
      margin: '2rem auto', 
      padding: '1.5rem', 
      border: '1px solid #e0e0e0', 
      borderRadius: 8,
      boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
    }}>
      <h2 style={{ color: '#2c3e50', marginBottom: '1rem' }}>SafeNest Hyderabad Navigation</h2>
      
      {/* Health status */}
      <div style={{ 
        padding: '0.5rem', 
        marginBottom: '1rem',
        backgroundColor: health ? '#e8f5e9' : '#ffebee',
        borderRadius: 4
      }}>
        {health ? (
          <div style={{ fontSize: 14, color: '#2e7d32' }}>
            <b>Backend Status:</b> {health.status.toUpperCase()} | 
            <b> Location:</b> {health.location} | 
            <b> Active Sessions:</b> {health.sessions} | 
            <b> Graph Nodes:</b> {health.graph_stats.nodes} | 
            <b> Uptime:</b> {Math.round(health.uptime)} seconds
          </div>
        ) : (
          <div style={{ fontSize: 14, color: '#c62828' }}>
            <b>Warning:</b> Backend service unavailable
          </div>
        )}
      </div>
      
      {/* Coordinates input */}
      <div style={{ 
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '1rem',
        marginBottom: '1rem'
      }}>
        <div>
          <h4 style={{ marginBottom: '0.5rem' }}>Current Location</h4>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <input
              type="number"
              name="current_lat"
              value={coords.current_lat}
              onChange={handleInput}
              step="0.000001"
              placeholder="Latitude"
              style={{ flex: 1, padding: '0.5rem' }}
            />
            <input
              type="number"
              name="current_lon"
              value={coords.current_lon}
              onChange={handleInput}
              step="0.000001"
              placeholder="Longitude"
              style={{ flex: 1, padding: '0.5rem' }}
            />
          </div>
        </div>
        
        <div>
          <h4 style={{ marginBottom: '0.5rem' }}>Destination</h4>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <input
              type="number"
              name="dest_lat"
              value={coords.dest_lat}
              onChange={handleInput}
              step="0.000001"
              placeholder="Latitude"
              style={{ flex: 1, padding: '0.5rem' }}
            />
            <input
              type="number"
              name="dest_lon"
              value={coords.dest_lon}
              onChange={handleInput}
              step="0.000001"
              placeholder="Longitude"
              style={{ flex: 1, padding: '0.5rem' }}
            />
          </div>
        </div>
      </div>
      
      {/* Optimization options */}
      <div style={{ marginBottom: '1rem' }}>
        <label style={{ display: 'block', marginBottom: '0.5rem' }}>
          <b>Optimization Criteria:</b>
        </label>
        <select 
          value={optimizeFor} 
          onChange={e => setOptimizeFor(e.target.value)}
          style={{ width: '100%', padding: '0.5rem' }}
        >
          {optimizationOptions.map(opt => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </div>
      
      {/* Action buttons */}
      <div style={{ 
        display: 'flex',
        gap: '0.5rem',
        marginBottom: '1rem'
      }}>
        <button 
          onClick={startNavigation} 
          disabled={loading}
          style={{
            flex: 1,
            padding: '0.5rem',
            backgroundColor: '#4caf50',
            color: 'white',
            border: 'none',
            borderRadius: 4,
            cursor: 'pointer'
          }}
        >
          {loading ? 'Loading...' : 'Start Navigation'}
        </button>
        
        <button 
          onClick={updatePosition} 
          disabled={loading || !sessionId}
          style={{
            flex: 1,
            padding: '0.5rem',
            backgroundColor: '#2196f3',
            color: 'white',
            border: 'none',
            borderRadius: 4,
            cursor: 'pointer',
            opacity: sessionId ? 1 : 0.5
          }}
        >
          Update Position
        </button>
        
        <button 
          onClick={endNavigation} 
          disabled={loading || !sessionId}
          style={{
            flex: 1,
            padding: '0.5rem',
            backgroundColor: '#f44336',
            color: 'white',
            border: 'none',
            borderRadius: 4,
            cursor: 'pointer',
            opacity: sessionId ? 1 : 0.5
          }}
        >
          End Navigation
        </button>
      </div>
      
      {/* Session info */}
      {sessionId && (
        <div style={{ 
          padding: '0.5rem',
          marginBottom: '1rem',
          backgroundColor: '#e3f2fd',
          borderRadius: 4
        }}>
          <b>Session ID:</b> {sessionId}
          {nextCheckIn && (
            <span style={{ float: 'right' }}>
              <b>Next check in:</b> {nextCheckIn} seconds
            </span>
          )}
        </div>
      )}
      
      {/* Messages */}
      {message && (
        <div style={{ 
          padding: '0.5rem',
          marginBottom: '1rem',
          backgroundColor: '#e8f5e9',
          color: '#2e7d32',
          borderRadius: 4
        }}>
          {message}
        </div>
      )}
      
      {error && (
        <div style={{ 
          padding: '0.5rem',
          marginBottom: '1rem',
          backgroundColor: '#ffebee',
          color: '#c62828',
          borderRadius: 4
        }}>
          <b>Error:</b> {error}
        </div>
      )}
      
      {/* Route information */}
      {routeData && (
        <div style={{ marginBottom: '1rem' }}>
          <div style={{ 
            padding: '1rem',
            border: '1px solid #e0e0e0',
            borderRadius: 4,
            marginBottom: '1rem'
          }}>
            <h3 style={{ marginBottom: '0.5rem' }}>Route Information</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div>
                <div><b>Distance:</b> {routeData.distance.toFixed(0)} meters</div>
                <div><b>Estimated Time:</b> {routeData.estimated_time.toFixed(1)} minutes</div>
              </div>
              <div>
                <div><b>Optimization:</b> {optimizationOptions.find(o => o.value === optimizeFor)?.label}</div>
                <div><b>Waypoints:</b> {routeData.waypoints.length}</div>
              </div>
            </div>
          </div>
          
          {/* Map display */}
          <div style={{ marginBottom: '1rem' }}>
            <h3 style={{ marginBottom: '0.5rem' }}>Route Map</h3>
            <MapDisplay 
              route={routeData.route} 
              currentPosition={currentPosition}
              destination={destination}
            />
          </div>
        </div>
      )}
      
      {/* Change destination */}
      {sessionId && (
        <div style={{ 
          padding: '1rem',
          border: '1px solid #e0e0e0',
          borderRadius: 4,
          marginBottom: '1rem'
        }}>
          <h3 style={{ marginBottom: '0.5rem' }}>Change Destination</h3>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <input
              type="number"
              value={coords.dest_lat}
              onChange={e => setCoords(prev => ({ ...prev, dest_lat: parseFloat(e.target.value) || 0 }))}
              step="0.000001"
              placeholder="New Latitude"
              style={{ flex: 1, padding: '0.5rem' }}
            />
            <input
              type="number"
              value={coords.dest_lon}
              onChange={e => setCoords(prev => ({ ...prev, dest_lon: parseFloat(e.target.value) || 0 }))}
              step="0.000001"
              placeholder="New Longitude"
              style={{ flex: 1, padding: '0.5rem' }}
            />
            <button 
              onClick={changeDestination}
              disabled={loading}
              style={{
                padding: '0 1rem',
                backgroundColor: '#ff9800',
                color: 'white',
                border: 'none',
                borderRadius: 4,
                cursor: 'pointer'
              }}
            >
              Change
            </button>
          </div>
        </div>
      )}
      
      {/* Incident reporting */}
      {sessionId && (
        <div style={{ 
          padding: '1rem',
          border: '1px solid #e0e0e0',
          borderRadius: 4
        }}>
          {!showReportForm ? (
            <button 
              onClick={() => setShowReportForm(true)}
              style={{
                width: '100%',
                padding: '0.5rem',
                backgroundColor: '#9c27b0',
                color: 'white',
                border: 'none',
                borderRadius: 4,
                cursor: 'pointer'
              }}
            >
              Report Safety Incident
            </button>
          ) : (
            <div>
              <h3 style={{ marginBottom: '0.5rem' }}>Report Safety Incident</h3>
              
              <div style={{ marginBottom: '0.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.25rem' }}>Incident Type:</label>
                <select
                  name="incident_type"
                  value={incidentData.incident_type}
                  onChange={handleIncidentInput}
                  style={{ width: '100%', padding: '0.5rem' }}
                >
                  {incidentTypes.map(type => (
                    <option key={type.value} value={type.value}>{type.label}</option>
                  ))}
                </select>
              </div>
              
              <div style={{ marginBottom: '0.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.25rem' }}>Severity:</label>
                <select
                  name="severity"
                  value={incidentData.severity}
                  onChange={handleIncidentInput}
                  style={{ width: '100%', padding: '0.5rem' }}
                >
                  {severityLevels.map(level => (
                    <option key={level.value} value={level.value}>{level.label}</option>
                  ))}
                </select>
              </div>
              
              <div style={{ marginBottom: '0.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.25rem' }}>Description:</label>
                <textarea
                  name="description"
                  value={incidentData.description}
                  onChange={handleIncidentInput}
                  placeholder="Describe the incident in detail"
                  style={{ width: '100%', padding: '0.5rem', minHeight: '80px' }}
                />
              </div>
              
              <div style={{ marginBottom: '0.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.25rem' }}>Your Contact (optional):</label>
                <input
                  type="text"
                  name="user_contact"
                  value={incidentData.user_contact}
                  onChange={handleIncidentInput}
                  placeholder="Phone/email for follow-up"
                  style={{ width: '100%', padding: '0.5rem' }}
                />
              </div>
              
              <div style={{ marginBottom: '0.5rem' }}>
                <label style={{ display: 'block', marginBottom: '0.25rem' }}>Incident Location:</label>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <input
                    type="number"
                    name="location_lat"
                    value={incidentData.location_lat}
                    onChange={handleIncidentInput}
                    step="0.000001"
                    placeholder="Latitude"
                    style={{ flex: 1, padding: '0.5rem' }}
                  />
                  <input
                    type="number"
                    name="location_lon"
                    value={incidentData.location_lon}
                    onChange={handleIncidentInput}
                    step="0.000001"
                    placeholder="Longitude"
                    style={{ flex: 1, padding: '0.5rem' }}
                  />
                </div>
              </div>
              
              <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem' }}>
                <button 
                  onClick={submitIncidentReport}
                  disabled={loading}
                  style={{
                    flex: 1,
                    padding: '0.5rem',
                    backgroundColor: '#9c27b0',
                    color: 'white',
                    border: 'none',
                    borderRadius: 4,
                    cursor: 'pointer'
                  }}
                >
                  {loading ? 'Submitting...' : 'Submit Report'}
                </button>
                
                <button 
                  onClick={() => setShowReportForm(false)}
                  style={{
                    flex: 1,
                    padding: '0.5rem',
                    backgroundColor: '#f44336',
                    color: 'white',
                    border: 'none',
                    borderRadius: 4,
                    cursor: 'pointer'
                  }}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default Navigation;
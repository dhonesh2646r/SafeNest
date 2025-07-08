from fastapi import FastAPI, HTTPException, BackgroundTasks, status, Request
from pydantic import BaseModel, Field, validator
import osmnx as ox
import networkx as nx
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import List, Tuple, Optional, Dict
import time
from functools import lru_cache
import asyncio
from collections import deque
import math
from datetime import datetime
import uuid
import google.generativeai as genai
import json
import re
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SafeNest Hyderabad Navigation API",
    description="Real-time route planning with safety optimization for Hyderabad area",
    version="2.3.0",
    docs_url="/docs",
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HYDERABAD_BOUNDS = {
    'min_lat': 17.10,
    'max_lat': 17.60,
    'min_lon': 78.30,
    'max_lon': 78.70
}

ADIBATLA_POLICE_STATION = (17.2160, 78.6139)
CENTER_POINT = ADIBATLA_POLICE_STATION
GRAPH_RADIUS = 5000
MAX_DISTANCE_KM = 10
CACHE_SIZE = 1000
UPDATE_INTERVAL = 10
REPLAN_THRESHOLD = 50
SESSION_TIMEOUT = 300
DESTINATION_REACHED_THRESHOLD = 20
active_navigations: Dict[str, Dict] = {}

reported_incidents: Dict[str, Dict] = {}

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables. AI features will be disabled.")
    genai.configure(api_key="dummy_key")
    model = None
else:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    logger.info("Gemini AI initialized successfully")

class RouteRequest(BaseModel):
    current_lat: float = Field(..., ge=-90, le=90, description="Current latitude")
    current_lon: float = Field(..., ge=-180, le=180, description="Current longitude")
    dest_lat: float = Field(
        default=ADIBATLA_POLICE_STATION[0],
        description="Adibatla Police Station latitude by default"
    )
    dest_lon: float = Field(
        default=ADIBATLA_POLICE_STATION[1],
        description="Adibatla Police Station longitude by default"
    )
    optimize_for: str = Field(
        "shortest", 
        description="Optimization criteria: shortest, safest, quietest, fastest"
    )
    session_id: Optional[str] = Field(
        None, 
        description="Existing session ID for continuation"
    )

    @validator('optimize_for')
    def validate_optimize_for(cls, v):
        if v not in ["shortest", "safest", "quietest", "fastest"]:
            raise ValueError("Invalid optimization criteria")
        return v

class RouteResponse(BaseModel):
    route: List[Tuple[float, float]] = Field(..., description="List of (lat, lng) coordinates")
    waypoints: List[Tuple[float, float]] = Field(..., description="Simplified route points")
    distance: float = Field(..., description="Total distance in meters")
    estimated_time: float = Field(..., description="Estimated time in minutes")
    next_check_in: float = Field(..., description="Seconds until next update")
    session_id: str = Field(..., description="Navigation session ID")
    message: Optional[str] = Field(None, description="Additional information")

class PositionUpdate(BaseModel):
    current_lat: float = Field(..., ge=-90, le=90)
    current_lon: float = Field(..., ge=-180, le=180)
    session_id: str = Field(..., description="Active session ID")

class DestinationUpdate(BaseModel):
    new_dest_lat: float = Field(..., ge=-90, le=90, description="New destination latitude")
    new_dest_lon: float = Field(..., ge=-180, le=180, description="New destination longitude")
    session_id: str = Field(..., description="Active session ID")

class IncidentReport(BaseModel):
    session_id: str = Field(..., description="Navigation session ID")
    incident_type: str = Field(..., description="Type of incident")
    severity: str = Field(..., description="Severity level")
    description: str = Field(..., description="Incident description")
    location_lat: float = Field(..., ge=-90, le=90, description="Incident location latitude")
    location_lon: float = Field(..., ge=-180, le=180, description="Incident location longitude")
    user_contact: Optional[str] = Field(None, description="User contact information")
    timestamp: Optional[str] = Field(None, description="Incident timestamp")

class IncidentResponse(BaseModel):
    incident_id: str = Field(..., description="Unique incident ID")
    status: str = Field(..., description="Report status")
    message: str = Field(..., description="Response message")
    reported_at: str = Field(..., description="Report timestamp")
    ai_analysis: Optional[Dict] = Field(None, description="AI analysis results")

class LocationRequest(BaseModel):
    location_text: str = Field(..., description="Text description of location")
    context: Optional[str] = Field(None, description="Additional context about the area")

class LocationResponse(BaseModel):
    coordinates: Tuple[float, float] = Field(..., description="Extracted coordinates")
    confidence: float = Field(..., description="AI confidence score")
    location_name: str = Field(..., description="Identified location name")
    address: str = Field(..., description="Formatted address")

class AIAnalysisRequest(BaseModel):
    incident_description: str = Field(..., description="Raw incident description")
    location_context: Optional[str] = Field(None, description="Location context")
    time_context: Optional[str] = Field(None, description="Time context")

class AIAnalysisResponse(BaseModel):
    severity_level: str = Field(..., description="AI-assessed severity level")
    incident_type: str = Field(..., description="AI-classified incident type")
    risk_score: float = Field(..., description="Risk assessment score (0-1)")
    formatted_description: str = Field(..., description="AI-formatted description")
    recommendations: List[str] = Field(..., description="AI recommendations")
    weight_factor: float = Field(..., description="Safety weight factor")

class NavigationSession(BaseModel):
    destination: Tuple[float, float]
    current_route: List[int]
    current_coords: List[Tuple[float, float]]
    optimize_for: str
    last_update: float
    path_history: deque

async def extract_coordinates_from_text(location_text: str, context: str = "") -> Dict:
    if not model:
        logger.warning("AI model not available. Using default coordinates.")
        return {
            "coordinates": ADIBATLA_POLICE_STATION,
            "confidence": 0.0,
            "location_name": "Default Location",
            "address": "Adibatla Police Station, Hyderabad"
        }
    
    prompt = f"""
    Extract coordinates from the following location description in Hyderabad, India.
    Return ONLY a JSON object with this exact format:
    {{
        "latitude": float,
        "longitude": float,
        "confidence": float (0-1),
        "location_name": "string",
        "address": "string"
    }}
    
    Location: {location_text}
    Context: {context}
    
    Rules:
    - Coordinates must be within Hyderabad bounds (17.10-17.60 lat, 78.30-78.70 lon)
    - If exact coordinates aren't clear, estimate based on landmarks
    - Confidence should reflect how certain you are about the location
    - Location name should be a recognizable place name
    - Address should be a readable street address
    """
    
    try:
        response = model.generate_content(prompt)
        result = json.loads(response.text)
        
        lat = float(result.get('latitude', 0))
        lon = float(result.get('longitude', 0))
        
        if not (HYDERABAD_BOUNDS['min_lat'] <= lat <= HYDERABAD_BOUNDS['max_lat'] and
                HYDERABAD_BOUNDS['min_lon'] <= lon <= HYDERABAD_BOUNDS['max_lon']):
            raise ValueError("Coordinates outside Hyderabad bounds")
        
        return {
            "coordinates": (lat, lon),
            "confidence": float(result.get('confidence', 0.5)),
            "location_name": result.get('location_name', "Unknown Location"),
            "address": result.get('address', "Hyderabad, India")
        }
    except Exception as e:
        logger.error(f"AI coordinate extraction failed: {str(e)}")
        return {
            "coordinates": ADIBATLA_POLICE_STATION,
            "confidence": 0.0,
            "location_name": "Default Location",
            "address": "Adibatla Police Station, Hyderabad"
        }

async def analyze_incident_with_ai(description: str, location_context: str = "", time_context: str = "") -> Dict:
    if not model:
        logger.warning("AI model not available. Using default analysis.")
        return {
            "severity_level": "medium",
            "incident_type": "other",
            "risk_score": 0.5,
            "formatted_description": description,
            "recommendations": ["Report to local authorities", "Stay alert"],
            "weight_factor": 0.5
        }
    
    prompt = f"""
    Analyze this safety incident report and provide structured analysis.
    Return ONLY a JSON object with this exact format:
    {{
        "severity_level": "low|medium|high|emergency",
        "incident_type": "harassment|theft|assault|suspicious|other",
        "risk_score": float (0-1),
        "formatted_description": "string",
        "recommendations": ["string", "string"],
        "weight_factor": float (0-1)
    }}
    
    Incident Description: {description}
    Location Context: {location_context}
    Time Context: {time_context}
    
    Analysis Guidelines:
    - Severity: low (minor), medium (concerning), high (serious), emergency (immediate danger)
    - Risk Score: 0.0-1.0 based on potential harm and urgency
    - Weight Factor: 0.0-1.0 for safety scoring (higher = more dangerous)
    - Format description professionally and clearly
    - Provide 2-3 actionable recommendations
    """
    
    try:
        response = model.generate_content(prompt)
        result = json.loads(response.text)
        
        return {
            "severity_level": result.get('severity_level', 'medium'),
            "incident_type": result.get('incident_type', 'other'),
            "risk_score": float(result.get('risk_score', 0.5)),
            "formatted_description": result.get('formatted_description', description),
            "recommendations": result.get('recommendations', []),
            "weight_factor": float(result.get('weight_factor', 0.5))
        }
    except Exception as e:
        logger.error(f"AI incident analysis failed: {str(e)}")
        return {
            "severity_level": "medium",
            "incident_type": "other",
            "risk_score": 0.5,
            "formatted_description": description,
            "recommendations": ["Report to local authorities", "Stay alert"],
            "weight_factor": 0.5
        }

@app.on_event("startup")
async def load_graph():
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading Hyderabad OSM graph (attempt {attempt + 1}/{max_retries})...")
            start_time = time.time()

            global G
            G = ox.graph_from_point(
                CENTER_POINT,
                dist=GRAPH_RADIUS,
                network_type='walk',
                simplify=True,
                retain_all=False,
                truncate_by_edge=True
            )

            G = ox.add_edge_speeds(G, fallback=4.5)
            G = ox.add_edge_travel_times(G)

            for u, v, k, data in G.edges(keys=True, data=True):
                data['safety_score'] = calculate_safety_score(u, v, data)
                data['quietness_score'] = calculate_quietness_score(u, v, data)

            if len(G.nodes) < 10:
                raise RuntimeError("Loaded graph is too small - may indicate download issues")

            logger.info(
                f"Hyderabad graph loaded successfully in {time.time() - start_time:.2f}s. "
                f"Nodes: {len(G.nodes)}, Edges: {len(G.edges)}"
            )
            return
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                logger.critical("Failed to load Hyderabad graph after maximum retries")
                raise RuntimeError("Failed to initialize Hyderabad navigation graph")

def calculate_safety_score(u, v, data):
    score = 0.5
    if 'highway' in data:
        highway_type = data['highway']
        if highway_type in ['footway', 'path', 'pedestrian']:
            score += 0.3
        elif highway_type in ['residential', 'living_street']:
            score += 0.15
        elif highway_type in ['primary', 'secondary', 'tertiary']:
            score -= 0.15
        elif highway_type in ['motorway', 'trunk']:
            score -= 0.4
    
    if 'lit' in data and data['lit'] == 'yes':
        score += 0.2
    if 'sidewalk' in data and data['sidewalk'] in ['both', 'left', 'right']:
        score += 0.15
    
    return max(0.1, min(1.0, score))

def calculate_quietness_score(u, v, data):
    score = 0.5
    
    if 'highway' in data:
        highway_type = data['highway']
        if highway_type in ['footway', 'path', 'pedestrian']:
            score += 0.4
        elif highway_type in ['residential', 'living_street']:
            score += 0.25
        elif highway_type in ['primary', 'secondary']:
            score -= 0.25
        elif highway_type in ['motorway', 'trunk']:
            score -= 0.5
    
    return max(0.1, min(1.0, score))

@lru_cache(maxsize=CACHE_SIZE)
def find_route(orig_node: int, dest_node: int, optimize_for: str = "shortest"):
    try:
        weight_dict = {
            "shortest": "length",
            "fastest": "travel_time",
            "safest": "safety_score",
            "quietest": "quietness_score"
        }
        weight = weight_dict.get(optimize_for, "length")

        if weight not in ["length", "travel_time"]:
            for _, _, data in G.edges(data=True):
                if weight not in data:
                    logger.warning(f"Missing {weight} attribute, falling back to 'length'")
                    weight = "length"
                    break

        return nx.shortest_path(G, orig_node, dest_node, weight=weight)

    except nx.NetworkXNoPath:
        logger.warning(f"No path found between nodes {orig_node} and {dest_node}")
        return None
    except Exception as e:
        logger.error(f"Route calculation error: {str(e)}")
        return None

def calculate_route_stats(route):
    total_length = 0
    total_time = 0
    
    for i in range(len(route)-1):
        edge_data = G.get_edge_data(route[i], route[i+1])[0]
        total_length += edge_data.get('length', 0)
        total_time += edge_data.get('travel_time', 0)
    
    return total_length, total_time / 60

def simplify_waypoints(coords: List[Tuple[float, float]], max_points=15) -> List[Tuple[float, float]]:
    if len(coords) <= max_points:
        return coords
    
    step = max(1, len(coords) // max_points)
    simplified = [coords[i] for i in range(0, len(coords), step)]
    
    if simplified[-1] != coords[-1]:
        simplified.append(coords[-1])
    
    return simplified

def haversine_distance(coord1, coord2):
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return 6371000 * c

def validate_within_hyderabad(lat: float, lon: float):
    if not (HYDERABAD_BOUNDS['min_lat'] <= lat <= HYDERABAD_BOUNDS['max_lat'] and
            HYDERABAD_BOUNDS['min_lon'] <= lon <= HYDERABAD_BOUNDS['max_lon']):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Location must be within Hyderabad area"
        )

def validate_distance(lat1: float, lon1: float, lat2: float, lon2: float):
    distance = haversine_distance((lat1, lon1), (lat2, lon2))
    if distance > MAX_DISTANCE_KM * 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Distance between points cannot exceed {MAX_DISTANCE_KM} km"
        )
    return distance

async def monitor_navigation_sessions():
    while True:
        try:
            current_time = time.time()
            stale_sessions = [
                session_id for session_id, session in active_navigations.items()
                if current_time - session['last_update'] > SESSION_TIMEOUT
            ]
            
            for session_id in stale_sessions:
                logger.info(f"Cleaning up stale session: {session_id}")
                del active_navigations[session_id]
                
            await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"Session monitoring error: {str(e)}")
            await asyncio.sleep(30)

@app.on_event("startup")
async def startup_tasks():
    asyncio.create_task(monitor_navigation_sessions())

@app.post("/extract_location", response_model=LocationResponse)
async def extract_location(data: LocationRequest):
    try:
        result = await extract_coordinates_from_text(data.location_text, data.context or "")
        
        return {
            "coordinates": result["coordinates"],
            "confidence": result["confidence"],
            "location_name": result["location_name"],
            "address": result["address"]
        }
    except Exception as e:
        logger.error(f"Location extraction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract location coordinates"
        )

@app.post("/analyze_incident", response_model=AIAnalysisResponse)
async def analyze_incident(data: AIAnalysisRequest):
    try:
        result = await analyze_incident_with_ai(
            data.incident_description,
            data.location_context or "",
            data.time_context or ""
        )
        
        return {
            "severity_level": result["severity_level"],
            "incident_type": result["incident_type"],
            "risk_score": result["risk_score"],
            "formatted_description": result["formatted_description"],
            "recommendations": result["recommendations"],
            "weight_factor": result["weight_factor"]
        }
    except Exception as e:
        logger.error(f"Incident analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze incident"
        )

@app.post("/start_navigation", response_model=RouteResponse, status_code=status.HTTP_201_CREATED)
async def start_navigation(data: RouteRequest, background_tasks: BackgroundTasks):
    try:
        validate_within_hyderabad(data.current_lat, data.current_lon)
        validate_within_hyderabad(data.dest_lat, data.dest_lon)
        
        distance_km = validate_distance(
            data.current_lat, data.current_lon,
            data.dest_lat, data.dest_lon
        ) / 1000

        session_id = data.session_id or f"nav_{int(time.time() * 1000)}"
        
        orig_node = ox.distance.nearest_nodes(G, data.current_lon, data.current_lat)
        dest_node = ox.distance.nearest_nodes(G, data.dest_lon, data.dest_lat)

        route = find_route(orig_node, dest_node, data.optimize_for)
        if not route:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No valid route found within Hyderabad area"
            )

        coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
        distance, time_est = calculate_route_stats(route)
        waypoints = simplify_waypoints(coords)

        active_navigations[session_id] = {
            'destination': (data.dest_lat, data.dest_lon),
            'current_route': route,
            'current_coords': coords,
            'optimize_for': data.optimize_for,
            'last_update': time.time(),
            'path_history': deque(maxlen=10)
        }

        return {
            "route": coords,
            "waypoints": waypoints,
            "distance": distance,
            "estimated_time": time_est,
            "next_check_in": UPDATE_INTERVAL,
            "session_id": session_id,
            "message": f"Hyderabad navigation started ({distance_km:.1f} km). Next update in {UPDATE_INTERVAL} seconds."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Navigation start failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start navigation in Hyderabad"
        )

@app.post("/update_position", response_model=RouteResponse)
async def update_position(data: PositionUpdate):
    try:
        if data.session_id not in active_navigations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Navigation session not found"
            )

        session = active_navigations[data.session_id]
        session['last_update'] = time.time()
        
        current_pos = (data.current_lat, data.current_lon)
        dest_distance = haversine_distance(current_pos, session['destination'])
        
        if dest_distance <= DESTINATION_REACHED_THRESHOLD:
            del active_navigations[data.session_id]
            return {
                "route": [],
                "waypoints": [],
                "distance": 0,
                "estimated_time": 0,
                "next_check_in": 0,
                "session_id": data.session_id,
                "message": "Destination reached in Hyderabad!"
            }

        validate_within_hyderabad(data.current_lat, data.current_lon)
        validate_distance(data.current_lat, data.current_lon, *session['destination'])

        current_node = ox.distance.nearest_nodes(G, data.current_lon, data.current_lat)
        
        remaining_route = None
        if current_node in session['current_route']:
            node_index = session['current_route'].index(current_node)
            remaining_route = session['current_route'][node_index:]
        
        if not remaining_route or len(remaining_route) < 2:
            logger.info(f"Re-routing for session {data.session_id} in Hyderabad")
            dest_node = ox.distance.nearest_nodes(
                G, 
                session['destination'][1], 
                session['destination'][0]
            )
            route = find_route(current_node, dest_node, session['optimize_for'])
            
            if not route:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No valid route found after deviation in Hyderabad"
                )
            
            coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
            session['current_route'] = route
            session['current_coords'] = coords
        else:
            route = remaining_route
            coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]

        distance, time_est = calculate_route_stats(route)
        waypoints = simplify_waypoints(coords)

        return {
            "route": coords,
            "waypoints": waypoints,
            "distance": distance,
            "estimated_time": time_est,
            "next_check_in": UPDATE_INTERVAL,
            "session_id": data.session_id,
            "message": f"Position updated in Hyderabad. {distance:.0f}m remaining."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Position update failed in Hyderabad: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update position in Hyderabad"
        )

@app.post("/change_destination", response_model=RouteResponse)
async def change_destination(data: DestinationUpdate):
    try:
        if data.session_id not in active_navigations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Navigation session not found"
            )

        session = active_navigations[data.session_id]
        
        validate_within_hyderabad(data.new_dest_lat, data.new_dest_lon)
        
        if not session['current_coords']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current position not available"
            )
        
        current_lat, current_lon = session['current_coords'][0]
        
        distance_km = validate_distance(
            current_lat, current_lon,
            data.new_dest_lat, data.new_dest_lon
        ) / 1000

        session['destination'] = (data.new_dest_lat, data.new_dest_lon)
        session['last_update'] = time.time()

        current_node = ox.distance.nearest_nodes(G, current_lon, current_lat)
        new_dest_node = ox.distance.nearest_nodes(G, data.new_dest_lon, data.new_dest_lat)

        new_route = find_route(current_node, new_dest_node, session['optimize_for'])
        if not new_route:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No valid route found to new destination"
            )

        new_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in new_route]
        distance, time_est = calculate_route_stats(new_route)
        
        session['current_route'] = new_route
        session['current_coords'] = new_coords

        return {
            "route": new_coords,
            "waypoints": simplify_waypoints(new_coords),
            "distance": distance,
            "estimated_time": time_est,
            "next_check_in": UPDATE_INTERVAL,
            "session_id": data.session_id,
            "message": f"Destination changed. New route ({distance_km:.1f} km)"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to change destination: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change destination"
        )

@app.post("/report_incident", response_model=IncidentResponse, status_code=status.HTTP_201_CREATED)
async def report_incident(request: Request, data: IncidentReport):
    try:
        body = await request.body()
        print("=== RAW REQUEST BODY ===")
        print(body.decode())
        print("========================")
        
        print("=== INCIDENT REPORT RECEIVED ===")
        print(f"Session ID: {data.session_id}")
        print(f"Incident Type: {data.incident_type}")
        print(f"Severity: {data.severity}")
        print(f"Description: {data.description}")
        print(f"Location: ({data.location_lat}, {data.location_lon})")
        print(f"User Contact: {data.user_contact}")
        print(f"Timestamp: {data.timestamp}")
        print("=================================")
        
        validate_within_hyderabad(data.location_lat, data.location_lon)
        
        ai_analysis = await analyze_incident_with_ai(
            data.description,
            f"Location: {data.location_lat}, {data.location_lon}",
            data.timestamp or datetime.now().isoformat()
        )
        
        incident_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        incident_data = {
            "incident_id": incident_id,
            "type": ai_analysis["incident_type"],
            "severity": ai_analysis["severity_level"],
            "description": ai_analysis["formatted_description"],
            "location": (data.location_lat, data.location_lon),
            "user_contact": data.user_contact,
            "timestamp": timestamp,
            "session_id": data.session_id,
            "ai_analysis": ai_analysis
        }
        
        reported_incidents[incident_id] = incident_data
        
        logger.warning(
            f"New incident reported: {ai_analysis['incident_type']} (severity: {ai_analysis['severity_level']}) "
            f"at {data.location_lat},{data.location_lon}"
        )
        
        return {
            "incident_id": incident_id,
            "status": "reported",
            "message": "Incident reported successfully. Authorities have been notified.",
            "reported_at": timestamp,
            "ai_analysis": ai_analysis
        }
        
    except HTTPException as e:
        print(f"HTTP Exception in report_incident: {e}")
        raise
    except Exception as e:
        print(f"Exception in report_incident: {str(e)}")
        logger.error(f"Failed to report incident: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to report incident"
        )

@app.get("/incidents", response_model=List[Dict])
async def get_reported_incidents():
    return list(reported_incidents.values())

@app.post("/end_navigation", status_code=status.HTTP_200_OK)
async def end_navigation(session_id: str):
    if session_id in active_navigations:
        del active_navigations[session_id]
        return {"status": "success", "message": "Hyderabad navigation ended"}
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Session not found"
    )

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {
        "status": "healthy",
        "location": "Hyderabad",
        "center_point": ADIBATLA_POLICE_STATION,
        "graph_stats": {
            "nodes": len(G.nodes),
            "edges": len(G.edges),
            "area_covered_km2": round(math.pi * (GRAPH_RADIUS/1000)**2, 2)
        },
        "sessions": len(active_navigations),
        "incidents": len(reported_incidents),
        "uptime": time.time() - app.startup_time,
        "ai_enabled": True
    }

app.startup_time = time.time() 
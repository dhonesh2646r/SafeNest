# SafeNest Hyderabad Navigation with AI

A comprehensive navigation application with AI-powered features for safety and route optimization in Hyderabad, India.

## 🚀 Features

### Core Navigation
- **Real-time route planning** with multiple optimization options
- **Hyderabad-focused** navigation with geographic bounds validation
- **Session management** for continuous navigation
- **Interactive map** with Leaflet integration
- **Position updates** and route recalculation

### AI-Powered Features
- **Location Extraction**: Convert text descriptions to coordinates
- **Incident Analysis**: AI-powered severity assessment and formatting
- **Smart Recommendations**: Automated safety recommendations
- **Risk Scoring**: Intelligent risk assessment for incidents

### Safety Features
- **Incident Reporting**: Comprehensive safety incident reporting
- **Multiple Severity Levels**: Low, Medium, High, Emergency
- **Contact Information**: Optional user contact for follow-up
- **Location Tracking**: Precise incident location recording

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 16+
- Gemini API key

### Backend Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Create environment file**:
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

3. **Get Gemini API Key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Add it to your `.env` file

4. **Start the backend server**:
   ```bash
   uvicorn route_planner:app --reload --host 127.0.0.1 --port 8000
   ```

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```

4. **Open in browser**:
   Navigate to `http://localhost:5173`

## 🎯 API Endpoints

### Navigation
- `POST /start_navigation` - Start navigation session
- `POST /update_position` - Update user position
- `POST /change_destination` - Change destination
- `POST /end_navigation` - End navigation session

### AI Features
- `POST /extract_location` - Extract coordinates from text
- `POST /analyze_incident` - Analyze incident with AI
- `POST /report_incident` - Report safety incident

### System
- `GET /health` - System health check
- `GET /incidents` - Get reported incidents

## 🤖 AI Integration

### Location Extraction
The AI can extract coordinates from natural language descriptions:
```
Input: "near Adibatla Police Station"
Output: Coordinates with confidence score
```

### Incident Analysis
AI automatically analyzes incident reports:
- **Severity Assessment**: Low, Medium, High, Emergency
- **Type Classification**: Harassment, Theft, Assault, etc.
- **Risk Scoring**: 0-100% risk assessment
- **Recommendations**: Actionable safety advice
- **Weight Factor**: Safety scoring for route optimization

## 📍 Hyderabad Coverage

The application is optimized for Hyderabad area:
- **Bounds**: 17.10°-17.60°N, 78.30°-78.70°E
- **Center**: Adibatla Police Station
- **Radius**: 5km coverage area
- **Max Distance**: 10km between points

## 🔧 Configuration

### Environment Variables
```env
GEMINI_API_KEY=your_api_key_here
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
```

### Backend Configuration
- **Graph Radius**: 5000m around center point
- **Cache Size**: 1000 route calculations
- **Update Interval**: 10 seconds
- **Session Timeout**: 300 seconds

## 🚨 Safety Features

### Incident Types
- Harassment
- Theft Attempt
- Assault
- Suspicious Activity
- Other Safety Concerns

### Severity Levels
- **Low**: Minor incidents
- **Medium**: Concerning situations
- **High**: Serious incidents
- **Emergency**: Immediate danger

## 📊 Route Optimization

### Optimization Criteria
- **Shortest**: Minimum distance
- **Fastest**: Minimum travel time
- **Safest**: Maximum safety score
- **Quietest**: Minimum noise level

### Safety Scoring
AI-enhanced safety scoring considers:
- Highway type (footway, residential, primary, etc.)
- Lighting conditions
- Sidewalk availability
- Incident history in area

## 🔍 Usage Examples

### Location Extraction
```
Input: "Cyberabad area near HITEC City"
Output: 
- Coordinates: 17.4456, 78.3772
- Location: HITEC City
- Confidence: 85.2%
```

### Incident Reporting
```
Input: "Someone following me near the bus stop"
AI Analysis:
- Severity: Medium
- Type: Suspicious Activity
- Risk Score: 65%
- Recommendations: [Stay in well-lit areas, Contact authorities]
```

## 🛡️ Security

- **Input Validation**: All coordinates validated against Hyderabad bounds
- **Distance Limits**: Maximum 10km between points
- **Session Management**: Automatic cleanup of stale sessions
- **Error Handling**: Comprehensive error responses

## 📈 Performance

- **Caching**: LRU cache for route calculations
- **Background Tasks**: Session monitoring
- **Optimized Queries**: Efficient OSM graph queries
- **Real-time Updates**: Minimal latency position updates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Review the health endpoint at `/health`
3. Check the logs for detailed error information

## 🔄 Updates

### Version 2.3.0
- Added AI-powered location extraction
- Enhanced incident analysis with Gemini
- Improved frontend with AI features
- Added comprehensive error handling

### Version 2.2.0
- Hyderabad-focused navigation
- Real-time route planning
- Safety incident reporting
- Interactive map interface 
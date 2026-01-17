# Backend Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                       │
│                    http://localhost:3000                         │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/REST API
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                         │
│                         (main.py)                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              CORS Middleware                                │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Routers    │    │   Schemas    │    │  Services    │
│              │    │              │    │              │
│ • customers  │◄───┤ • customer   │    │ • jar_svc    │
│ • sales      │    │ • sale       │    │ • summary    │
│ • expenses   │    │ • expense    │    │              │
│ • jars       │    │ • jar        │    └──────┬───────┘
│ • payments   │    │ • reminder   │           │
│ • reminders  │    │              │           │
│ • dashboard  │    └──────────────┘           │
└──────┬───────┘                               │
       │                                        │
       │         ┌──────────────┐              │
       │         │    Utils     │              │
       └────────►│              │◄─────────────┘
                 │ • helpers    │
                 │ • normalize  │
                 └──────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │ Dependencies │
                 │              │
                 │ • get_db()   │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │  Database    │
                 │              │
                 │ • engine     │
                 │ • session    │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │   Models     │
                 │              │
                 │ • Customer   │
                 │ • Sale       │
                 │ • Expense    │
                 │ • JarTrack   │
                 │ • Payment    │
                 │ • Reminder   │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │  PostgreSQL  │
                 │   Database   │
                 └──────────────┘
```

## Request Flow

### Example: Create Sale

```
1. Frontend sends POST /sales
   ↓
2. FastAPI receives request
   ↓
3. CORS middleware validates origin
   ↓
4. Router (sales.py) handles endpoint
   ↓
5. Schema (SaleCreate) validates request body
   ↓
6. Dependencies inject database session
   ↓
7. Router calls business logic
   ↓
8. Service (jar_service) updates jar tracking
   ↓
9. Models (Sale, JarTracking) interact with DB
   ↓
10. Response serialized and returned
    ↓
11. Frontend receives JSON response
```

## Module Responsibilities

### 1. Routers Layer
**Purpose**: Handle HTTP requests and responses

**Responsibilities**:
- Define API endpoints
- Parse request parameters
- Call business logic
- Format responses
- Handle HTTP errors

**Example**:
```python
@router.post("/sales")
def create_sale(payload: SaleCreate, db: Session = Depends(get_db)):
    # Validate, process, return response
```

### 2. Schemas Layer
**Purpose**: Data validation and serialization

**Responsibilities**:
- Define request/response models
- Validate input data
- Type checking
- Auto-generate API docs
- Custom validators

**Example**:
```python
class SaleCreate(BaseModel):
    total_jars: int = Field(..., gt=0)
    cost_per_jar: Optional[float] = None
```

### 3. Services Layer
**Purpose**: Business logic and calculations

**Responsibilities**:
- Complex calculations
- Data transformations
- Multi-model operations
- Reusable business rules

**Example**:
```python
def recalc_jartracking(db, customer_id, customer_name):
    # Calculate jar inventory
```

### 4. Models Layer
**Purpose**: Database schema definition

**Responsibilities**:
- Define table structure
- Relationships between tables
- Database constraints
- ORM mappings

**Example**:
```python
class Sale(Base):
    __tablename__ = "sales"
    id = Column(Integer, primary_key=True)
```

### 5. Utils Layer
**Purpose**: Helper functions

**Responsibilities**:
- String formatting
- Date/time utilities
- Common transformations
- Reusable helpers

**Example**:
```python
def normalize_name(name: str) -> str:
    return name.strip().title()
```

## Data Flow Patterns

### Pattern 1: Simple CRUD
```
Router → Schema → Model → Database
```

### Pattern 2: Complex Business Logic
```
Router → Schema → Service → Model → Database
                    ↓
                  Utils
```

### Pattern 3: Multi-Model Transaction
```
Router → Schema → Service → Model 1 → Database
                    ↓
                  Model 2 → Database
                    ↓
                  Model 3 → Database
```

## Error Handling Flow

```
Exception Raised
    ↓
Service/Router catches
    ↓
HTTPException created
    ↓
FastAPI error handler
    ↓
JSON error response
    ↓
Frontend displays error
```

## Database Transaction Pattern

```python
# 1. Get database session
db = get_db()

try:
    # 2. Perform operations
    db.add(new_record)
    db.commit()
    
    # 3. Refresh to get updated data
    db.refresh(new_record)
    
    # 4. Return success
    return new_record
    
except Exception as e:
    # 5. Rollback on error
    db.rollback()
    raise HTTPException(...)
    
finally:
    # 6. Close session
    db.close()
```

## Security Layers

```
┌─────────────────────────────────────┐
│         CORS Middleware             │  ← Origin validation
├─────────────────────────────────────┤
│      Request Validation             │  ← Pydantic schemas
├─────────────────────────────────────┤
│      Business Logic                 │  ← Authorization checks
├─────────────────────────────────────┤
│      Database Layer                 │  ← SQL injection prevention
└─────────────────────────────────────┘
```

## Performance Considerations

### 1. Database Connection Pooling
- SQLAlchemy manages connection pool
- Reuses connections efficiently
- Configurable pool size

### 2. Query Optimization
- Use joins instead of multiple queries
- Filter at database level
- Index frequently queried columns

### 3. Response Optimization
- Return only necessary fields
- Paginate large result sets
- Use database aggregations

## Scalability Path

### Current: Monolithic
```
Single FastAPI instance → Single PostgreSQL
```

### Future: Microservices
```
API Gateway
    ↓
┌───────┬───────┬───────┐
│ Sales │ Cust  │ Remind│  ← Separate services
└───┬───┴───┬───┴───┬───┘
    ↓       ↓       ↓
   DB1     DB2     DB3      ← Separate databases
```

## Testing Strategy

### Unit Tests
```
Test individual functions in:
- Services
- Utils
- Schemas validators
```

### Integration Tests
```
Test router endpoints:
- Mock database
- Test full request/response cycle
```

### End-to-End Tests
```
Test complete flows:
- Real database
- Multiple endpoints
- Business scenarios
```

## Deployment Architecture

```
┌─────────────────────────────────────┐
│         Heroku Platform             │
│  ┌───────────────────────────────┐  │
│  │   FastAPI App (Uvicorn)       │  │
│  │   - main.py                   │  │
│  │   - All routers loaded        │  │
│  └───────────────┬───────────────┘  │
│                  │                   │
│                  ▼                   │
│  ┌───────────────────────────────┐  │
│  │   PostgreSQL Database         │  │
│  │   - Managed by Heroku         │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

## Configuration Management

```
Environment Variables (.env)
    ↓
database.py reads DATABASE_URL
    ↓
Creates engine and session
    ↓
Used by all routers via dependency injection
```

## Best Practices Implemented

1. ✅ **Separation of Concerns**: Each module has single responsibility
2. ✅ **DRY Principle**: Reusable services and utils
3. ✅ **Type Safety**: Pydantic schemas for validation
4. ✅ **Dependency Injection**: Database sessions injected
5. ✅ **Error Handling**: Consistent HTTPException usage
6. ✅ **RESTful Design**: Standard HTTP methods and status codes
7. ✅ **Documentation**: Auto-generated OpenAPI docs
8. ✅ **Modularity**: Easy to add/remove features
9. ✅ **Testability**: Clear boundaries for testing
10. ✅ **Maintainability**: Organized file structure

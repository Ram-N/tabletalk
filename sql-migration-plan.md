# SQL Agent Migration Implementation Plan

## Overview
Migrate from LangChain experimental CSV agent to SQL Q&A agent to achieve 10-20x performance improvement (from 1-5 minutes to 5-10 seconds response time).

## Phase 1: Project Setup & Database Creation (Day 1, Morning)

### 1.1 Version Control Setup
- Create feature branch: `git checkout -b feature/sql-agent-migration`
- Ensure clean working directory before starting

### 1.2 Dependencies Installation
```bash
uv pip install sqlalchemy sqlite3 pandas
# Update requirements.txt
```

### 1.3 Database Schema Design
**Student Table Structure:**
- **Demographics**: student_id (PK), student_name, student_age, student_gender, school_name, city, grade_level
- **Categories**: student_category (ATS 2024, Sponsored, Full/Partial Gold Scholarship, Returning, Insider Circle, GW Family, Schools, Other Contacts)
- **Program**: program_name, track_chosen (Explorers/Wizards/Pathfinders), courses_selected
- **Courses**: course_1, instructor_1, course_2, instructor_2, course_3, instructor_3, teaching_assistant
- **Admin**: weeks_attending, payment_status, rc_name
- **Parents**: parent_name, parent_phone_primary, parent_phone_secondary, parent_email

### 1.4 Database Creation Script
- Create `database/` directory
- Build CSV → SQLite import script with data validation
- Add proper column types, indexes, and constraints
- Handle data quality issues (empty values, type conversion)

## Phase 2: SQL Agent Implementation (Day 1, Afternoon)

### 2.1 Replace CSV Agent Core
**File: `main.py`**
- Replace `create_csv_agent` import with `create_sql_agent`
- Update agent creation function to use database connection
- Configure read-only database permissions
- Add connection pooling for performance

### 2.2 Agent Configuration
- **Chain approach**: Fast, deterministic queries for simple questions
- **Agent approach**: Flexible reasoning for complex analysis
- Custom prompts optimized for student data domain
- Query result formatting and validation

### 2.3 API Endpoint Updates
- Modify `/upload` endpoint to import CSV → SQLite instead of direct processing
- Update both `/ask` and `/ask-stream` endpoints
- Add `/schema` endpoint for database introspection
- Maintain existing streaming functionality with new agent

## Phase 3: Performance Optimization (Day 2, Morning)

### 3.1 Database Optimization
- Add strategic indexes for common query patterns:
  - `student_category`, `city`, `school_name`, `track_chosen`
  - `grade_level`, `program_name`, `rc_name`
- Implement query result caching (in-memory for 654 rows)
- Add query execution time logging

### 3.2 Query Intelligence
- Pre-built SQL templates for common questions:
  - Student counts by category/city/school
  - Scholarship breakdowns
  - Course enrollment statistics
- Query complexity limits for safety
- Automatic query optimization suggestions

## Phase 4: Sample Queries Optimization (Day 2, Afternoon)

### 4.1 Priority Query Templates
Based on your sample queries, optimize for:
1. **Count queries**: "Total students in 2025", "Sponsored students count"
2. **Grouping queries**: "Top 5 schools", "Gender breakup", "Track breakup"
3. **Filtering queries**: "TVS schools students", "Full scholarship recipients"
4. **Multi-year analysis**: "PSBB students over years", "Parents with multiple children"
5. **Instructor queries**: "Courses taught by [instructor] over years"

### 4.2 Query Result Formatting
- Tabular format for lists and breakdowns
- Summary statistics for count queries
- Historical trend formatting for multi-year data
- Export functionality (CSV/JSON) for query results

## Phase 5: Frontend Enhancements (Day 3, Morning)

### 5.1 UI Updates
**File: `index.html`**
- Add "Database Schema" info panel showing table structure
- Update query examples with student data specific samples
- Add query result visualization (simple tables/charts)
- Improve error messages for SQL-specific errors

### 5.2 User Experience
- Query suggestion dropdown based on common patterns
- Auto-complete for column names and values
- Query history and favorites
- Real-time query validation feedback

## Phase 6: Testing & Validation (Day 3, Afternoon)

### 6.1 Performance Benchmarking
- Compare response times: CSV agent vs SQL agent
- Load testing with concurrent users
- Memory usage analysis
- Query complexity vs performance metrics

### 6.2 Data Accuracy Testing
- Validate query results against known data
- Test all sample queries you provided
- Edge case testing (empty results, invalid queries)
- Cross-reference with original CSV data

### 6.3 Security Testing
- SQL injection prevention validation
- Database permission restrictions testing
- Query complexity limits verification
- Input sanitization testing

## Phase 7: Documentation & Deployment (Day 4)

### 7.1 Code Documentation
- Update CLAUDE.md with new architecture details
- API documentation for new endpoints
- Database schema documentation
- Query examples and best practices

### 7.2 Deployment Preparation
- Database backup/restore procedures
- Migration rollback plan (keep CSV agent as fallback)
- Performance monitoring setup
- Error logging and alerting

## Risk Mitigation Strategies

### 1. Fallback Plan
- Keep existing CSV agent code intact during transition
- Feature flag to switch between agents
- Automated fallback on SQL agent failures

### 2. Data Integrity
- Comprehensive data validation during CSV import
- Backup original CSV files
- Transaction-based database operations
- Data consistency checks

### 3. Performance Safeguards
- Query timeout limits (30 seconds max)
- Connection pooling limits
- Memory usage monitoring
- Graceful degradation for complex queries

## Expected Outcomes

### Performance Metrics
- **Response Time**: 5-10 seconds (vs current 1-5 minutes)
- **Concurrency**: Support 5-10 simultaneous users
- **Memory Usage**: <100MB for 654 rows
- **Query Success Rate**: >95% for valid questions

### User Experience Improvements
- Real-time query results
- Better error messages
- Query suggestions and auto-complete
- Historical query analysis capabilities

### Maintainability Benefits
- Easier debugging with SQL query logs
- Standard database optimization techniques
- Better scalability for larger datasets
- Cleaner separation of concerns

## Student Data Categories Reference

### Student Categories (student_category field):
1. **ATS 2024** (42 students) - Students admitted based on 2024 Admissions Test Score data
2. **Other Contacts** (21 students) - Students from other referral sources
3. **Sponsored** (15 students) - Students with means-based scholarships
4. **Full Gold Scholarship** (14 students) - Students with full merit scholarships
5. **Returning** (10 students) - Students returning from previous years
6. **Partial Gold Scholarship** (10 students) - Students with partial merit scholarships
7. **Insider Circle** (10 students) - Students who signed up in advance for the program
8. **GW Family** (4 students) - GenWise family connections
9. **Schools** (3 students) - School-based referrals

### Common Query Patterns:
- "What were the total number of students who attended our program in 2025 May?"
- "How many sponsored students in 2025?"
- "What were the top 5 schools in terms of number of students?"
- "Who were the RCs in 2025?"
- "How many students came on a full scholarship in 2025? What were their names?"
- "How many students were from the TVS schools in 2025?"
- "Breakup of 2025 participants - by Explorers, Wizards, Pathfinders"
- "Gender breakup of the 2025 cohort"
- "Students from PSBB over the years"
- "Parents who have sent the most children to our Programs over the years"
- "Names of sponsored students in 2024"
- "Breakup of students by Course in 2025"
- "Courses taught by Utpal over the years"

## Success Criteria
✅ All sample queries execute in <10 seconds  
✅ Existing functionality maintained (streaming, file upload, etc.)  
✅ Database handles all current data without issues  
✅ Performance improvement of 10-20x demonstrated  
✅ Production-ready security and error handling  

This plan transforms your application from a slow, code-generating CSV agent into a fast, reliable SQL-powered analytics tool while maintaining all existing features and user experience.
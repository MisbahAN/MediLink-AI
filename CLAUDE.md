# CLAUDE.md - AI Assistant Guidelines for MediLink-AI Project

## Project Overview

This document provides guidelines for AI assistants working on the Prior Authorization automation project. Follow these conventions to ensure consistent, high-quality code generation.

## Code Style Guidelines

### General Conventions

- Follow standard language conventions (PEP 8 for Python, ESLint defaults for JS/TS)
- Maintain consistent indentation (4 spaces for Python, 2 spaces for JS/TS)
- Keep code clean and properly formatted
- Ensure all brackets, parentheses, and quotes match

### Documentation Standards

#### Python Functions/Methods

```python
def extract_patient_data(pdf_content: bytes) -> dict:
    """
    Extract patient information from PDF content.

    Args:
        pdf_content: Raw PDF file content as bytes

    Returns:
        Dictionary containing patient data with confidence scores
    """
    # Complex regex pattern for SSN extraction
    ssn_pattern = r'^\d{3}-\d{2}-\d{4}$'  # Matches XXX-XX-XXXX format
```

#### JavaScript/TypeScript Functions

```typescript
/**
 * Process uploaded files and send to backend
 * @param files - Array of PDF files to process
 * @returns Promise with upload response
 */
const uploadFiles = async (files: File[]): Promise<UploadResponse> => {
  // Complex validation logic
  const isValid = files.every(f => f.type === 'application/pdf');  // Ensure all files are PDFs
```

### Error Handling

- Mix approach based on criticality
- Use try-catch for external API calls and file operations
- Early returns for validation
- Let non-critical errors bubble to global handlers

### Logging

- Structured logging with context information
- Log important operations at INFO level
- Include request IDs and session IDs where applicable
- Use clear, searchable log messages

## Code Structure Preferences

### Function/Method Design

- Whatever length makes sense for clarity
- Follow single responsibility principle
- Extract complex logic into helper functions
- Prioritize readability over brevity

### Class Design

- Functional approach where it makes sense
- Use classes for services and complex state management
- Keep interfaces clean and focused
- Follow dependency injection patterns in FastAPI

## Naming Conventions

### Variables and Functions

- **Python**: `snake_case` for variables and functions
- **JavaScript/TypeScript**: `camelCase` for variables and functions
- **Constants**: `UPPER_SNAKE_CASE` in both languages
- Use descriptive names that explain purpose

### Files

- **Python files**: `snake_case.py`
- **React components**: `PascalCase.tsx`
- **Other JS/TS files**: `camelCase.ts`
- **Config files**: `lowercase.config.js`

## Testing Guidelines

### Test Implementation

- Write tests after implementing features
- Focus on critical paths and edge cases
- Use descriptive test names

### Test Naming

```python
def test_should_extract_patient_name_when_pdf_is_valid():
    """Test that patient name is correctly extracted from valid PDF"""
    pass
```

## AI Assistant Behavior

### Response Format

1. **Code first** - Provide complete, working code
2. **Brief explanation after** - Explain key design decisions
3. **Inline comments** - Only for complex logic

### Task Management

- **Always review CLAUDE.md before starting**
- **Complete exactly 3 tasks from todo.md at once** (unless specified otherwise)
- **After each file generation:**
  - Update todo.md for the task you completed
  - Update progress_notes.md with what was actually built (max 8 words).

### Example Response Pattern

```python
# [Complete working code here]
```

Mark `backend/app/services/pdf_extractor.py` as DONE in todo.md.

Update progress_notes.md with what was actually built (max 8 words).

### progress_notes.md

- pdf_extractor.py — base PDF extraction with coordinate support

## Technology-Specific Guidelines

### FastAPI Patterns

- Use dependency injection for services
- Implement service layer pattern
- Keep routes thin, logic in services
- Use Pydantic for all data validation

### React/Next.js Patterns

- Server components by default in Next.js 14
- Client components only when needed (interactivity)
- Custom hooks for reusable logic
- Proper error boundaries

## Performance Guidelines

- Balance readability and performance
- Optimize only when metrics show need
- Use async/await properly
- Implement caching strategically

## Always Do

- Include proper type hints in Python
- Add TypeScript types (never use `any`)
- Handle loading states in UI components
- Validate inputs at boundaries
- Include error messages user can understand
- Follow the project structure exactly
- Test with the actual test PDFs provided

## Never Do

- Skip error handling
- Use `any` or `unknown` types without reason
- Hardcode sensitive values
- Use inline styles in React components
- Ignore file size limits
- Make assumptions about PDF structure
- Skip validation on user inputs

## Communication Style

- Be concise but complete
- Assume intermediate programming knowledge
- Explain complex algorithms or patterns
- Focus on what was built, not how

## Progress Tracking Example

After building three files:

Mark `backend/app/api/routes/upload.py` as DONE in todo.md.
Mark `backend/app/services/storage.py` as DONE in todo.md.
Mark `backend/app/models/schemas.py` as DONE in todo.md.

Update progress_notes.md with what was actually built (max 8 words).

### progress_notes.md

- upload.py — PDF upload endpoint with validation logic
- storage.py — file storage service with session management
- schemas.py — Pydantic models for API data validation

# InfiniMetrics Project Guidelines

## Code Style

- **File line limit**: Single Python file should not exceed 300 lines. Split into multiple modules when needed.
- **Avoid duplicate code**: Check for existing reusable implementations before writing new code.
- **Code reuse**: Prioritize existing utility functions and classes. Don't reinvent the wheel.

## File Structure

```
infinimetrics/
├── common/          # Common utilities and constants
├── db/              # MongoDB related modules
│   ├── client.py    # Connection management
│   ├── repository.py # Data operations
│   └── ...
├── hardware/        # Hardware tests
├── operators/       # Operator tests
├── inference/       # Inference tests
└── communication/   # Communication tests

dashboard/
├── components/      # UI components
├── pages/           # Pages
└── utils/           # Utility functions
```

## Test Data Structure

- **Summary results**: `summary_output/dispatcher_summary_*.json` - Contains references to individual test result files
- **Individual results**: `output/*_results.json` - Contains detailed test data and CSV paths
- **CSV data**: Same directory as JSON, contains timeseries metric data

## MongoDB Collections

- `test_runs`: Stores test results with embedded CSV data

## Naming Conventions

- File names: snake_case
- Class names: PascalCase
- Functions/variables: snake_case

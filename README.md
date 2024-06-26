# Interactive System

## Setup Instructions

```
git clone https://github.com/ngquangtrung57/Interactive-System
cd Interactive_System
```

Create a .env file in the root directory and add your environment variables. The .env file should include:

```bash
OPENAI_API_KEY=your_openai_api_key
```

## Running without Docker
Install the required Python packages:
```bash
pip install -r requirements.txt
```
Run the Streamlit application:
```bash
streamlit run main.py
```

### Or using Docker
#### Prerequisites

Before setting up the bot, ensure you have the following installed:

1. Docker
2. Docker Compose

Use Docker Compose to build and start the Docker containers:

```bash
docker-compose up --build
```


### License
This project is licensed under the MIT License.

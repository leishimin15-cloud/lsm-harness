FROM python:3.13-slim

# Essential tools for a coding agent
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# Node.js (for npx, npm, MCP servers)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Safe defaults
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
WORKDIR /workspace

# Don't run as root scripts by default (can be overridden)
CMD ["sleep", "infinity"]

FROM python:latest
MAINTAINER Mumtaz Ahmad <mumtaz.ahmad@siemens.com>

ARG BRANCH=defaultValue
ARG COMMIT=defaultValue
ARG CODE_REPOSITORY=defaultValue

RUN echo "Building for $COMMIT"

RUN pip install termcolor
RUN pip install requests
RUN pip install pandas
RUN pip install plotly
RUN pip install matplotlib
RUN mkdir -p /src
RUN git clone $CODE_REPOSITORY --depth=1 --branch $BRANCH --single-branch src
RUN git config --global --add safe.directory /app




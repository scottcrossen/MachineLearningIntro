FROM ubuntu:16.04

RUN groupadd -r user --gid="1000" \
    && adduser --gid "1000" --uid "1000" --home "/home/user" --gecos '' --shell "/bin/bash" --disabled-password --disabled-login user \
    && chown -R user /home/user

RUN apt-get update \
  && apt-get install -y software-properties-common \
  && add-apt-repository -y ppa:webupd8team/java \
  && apt-get update \
  && echo 'yes' | apt-get install -y oracle-java8-installer \
  && apt-get install -y inotify-tools \
  && rm -rf /var/lib/apt/lists/* \
  && rm -rf /var/cache/oracle-jdk8-installer

COPY docker-entrypoint.sh /

WORKDIR "/"
USER user

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD []

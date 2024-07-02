ARG ROS_DISTRO=rolling
FROM ghcr.io/apl-ocean-engineering/blue:$ROS_DISTRO-robot as ci

# Non-root user from the upstream "blue" image
ARG USERNAME=ubuntu
ENV DEBIAN_FRONTEND=noninteractive

# WORKDIR /root/ws_blue
# COPY . src/blue

# Install apt packages needed for CI
USER root

#
# Don't require pinned package versions
#
# hadolint ignore=DL3008
RUN apt-get -q update \
    && apt-get -q -y upgrade \
    && apt-get -q install --no-install-recommends -y \
        gosu \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    # Test gosu
    && gosu nobody true
USER $USERNAME

FROM ci as deploy

ARG USERNAME
ENV DEBIAN_FRONTEND=noninteractive

# Switch to the non-root user for the rest of the installation
USER $USERNAME
ENV USER=$USERNAME

# Python in Ubuntu is now marked as a "Externally managed environment",
# Per best practice, create a venv for local python packages
#
# These two ENVs effectively "activate" the venv for subsequent calls to
# python/pip within this Dockerfile
#WORKDIR /home/$USERNAME
# ENV VIRTUAL_ENV=/home/$USERNAME/.venv/oakd_bluerov
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# # Venv needs to be established in an environment which already contains the ROS paths
# RUN ". /opt/ros/${ROS_DISTRO}/setup.sh && python3 -m venv --system-site-packages $VIRTUAL_ENV"

# RUN printf "source /opt/ros/%s/setup.bash\n" \
#             "source /%s/bin/activate" "${ROS_DISTRO}" "${VIRTUAL_ENV}}" >> /home/$USERNAME/.bashrc

ENV USER_WORKSPACE=/home/$USERNAME/ws_blue
WORKDIR $USER_WORKSPACE
COPY --chown=$USER_UID:$USER_GID . src/bluerov_fiducial_localization

# # Install the Python requirements that aren't available as rosdeps
# RUN python3 -m pip install --no-cache-dir -r ${USER_WORKSPACE}/src/oakd_bluerov/requirements.txt

USER root
RUN apt-get -q update \
    && gosu "${USER}" rosdep update \
    && gosu "${USER}" rosdep install -y --from-paths src --ignore-src --as-root=pip:false --rosdistro "${ROS_DISTRO}" \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf "/var/lib/apt/lists/*"
USER $USER

RUN . "/opt/ros/${ROS_DISTRO}/setup.sh" \
    && colcon build \
    && printf "source /%s/install/setup.bash"  "${USER_WORKSPACE}">> /home/${USERNAME}/.bashrc

ENTRYPOINT ["/usr/bin/bash"]

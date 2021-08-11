# Copyright Google Inc. 2017
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM python:3

ENV BK_VERSION=2.3.0
ENV PY_VERSION=3.7

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

# [START CMD]
#CMD bokeh serve --disable-index-redirect --num-procs=4 --port=8080 --address=0.0.0.0 --allow-websocket-origin=$LC_CIRCUIT_SIM LC_circuit_bokeh.py --allow-websocket-origin="https://quantum-circuits-307315.uc.r.appspot.com/Classical_LC"
CMD python wavfun_visualization.py
# [END CMD]
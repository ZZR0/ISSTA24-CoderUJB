./scripts/run_code_ujb.sh local_gen complete codeujbcomplete ./models/codellama-7b/ codellama-7b
./scripts/run_code_ujb.sh local_gen complete codeujbtestgen ./models/codellama-7b/ codellama-7b
./scripts/run_code_ujb.sh local_gen complete codeujbtestgenissue ./models/codellama-7b/ codellama-7b
./scripts/run_code_ujb.sh local_gen complete codeujbrepair ./models/codellama-7b/ codellama-7b
./scripts/run_code_ujb.sh local_gen complete codeujbdefectdetection ./models/codellama-7b/ codellama-7b

./scripts/run_code_ujb.sh local_gen chat codeujbcomplete ./models/codellama-instruct-7b/ codellama-instruct-7b
./scripts/run_code_ujb.sh local_gen chat codeujbtestgen ./models/codellama-instruct-7b/ codellama-instruct-7b
./scripts/run_code_ujb.sh local_gen chat codeujbtestgenissue ./models/codellama-instruct-7b/ codellama-instruct-7b
./scripts/run_code_ujb.sh local_gen chat codeujbrepair ./models/codellama-instruct-7b/ codellama-instruct-7b
./scripts/run_code_ujb.sh local_gen chat codeujbdefectdetection ./models/codellama-instruct-7b/ codellama-instruct-7b

# Please setting the Openai API key
# export OPENAI_API_BASE=''
# export OPENAI_API_KEY=''
./scripts/run_code_ujb.sh api_gen chat codeujbcomplete gpt-3.5-turbo gpt-3.5-turbo
./scripts/run_code_ujb.sh api_gen chat codeujbtestgen gpt-3.5-turbo gpt-3.5-turbo
./scripts/run_code_ujb.sh api_gen chat codeujbtestgenissue gpt-3.5-turbo gpt-3.5-turbo
./scripts/run_code_ujb.sh api_gen chat codeujbrepair gpt-3.5-turbo gpt-3.5-turbo
./scripts/run_code_ujb.sh api_gen chat codeujbdefectdetection gpt-3.5-turbo gpt-3.5-turbo
./scripts/run_code_ujb.sh api_gen chat lcb-gc-v1 gpt-3.5-turbo gpt-3.5-turbo

### 4. Evaluation
./scripts/run_code_ujb.sh eval complete codeujbcomplete ./models/codellama-7b/ codellama-7b
./scripts/run_code_ujb.sh eval complete codeujbtestgen ./models/codellama-7b/ codellama-7b
./scripts/run_code_ujb.sh eval complete codeujbtestgenissue ./models/codellama-7b/ codellama-7b
./scripts/run_code_ujb.sh eval complete codeujbrepair ./models/codellama-7b/ codellama-7b
./scripts/run_code_ujb.sh eval complete codeujbdefectdetection ./models/codellama-7b/ codellama-7b

./scripts/run_code_ujb.sh eval chat codeujbcomplete ./models/codellama-instruct-7b/ codellama-instruct-7b
./scripts/run_code_ujb.sh eval chat codeujbtestgen ./models/codellama-instruct-7b/ codellama-instruct-7b
./scripts/run_code_ujb.sh eval chat codeujbtestgenissue ./models/codellama-instruct-7b/ codellama-instruct-7b
./scripts/run_code_ujb.sh eval chat codeujbrepair ./models/codellama-instruct-7b/ codellama-instruct-7b
./scripts/run_code_ujb.sh eval chat codeujbdefectdetection ./models/codellama-instruct-7b/ codellama-instruct-7b

./scripts/run_code_ujb.sh eval chat codeujbcomplete gpt-3.5-turbo gpt-3.5-turbo
./scripts/run_code_ujb.sh eval chat codeujbtestgen gpt-3.5-turbo gpt-3.5-turbo
./scripts/run_code_ujb.sh eval chat codeujbtestgenissue gpt-3.5-turbo gpt-3.5-turbo
./scripts/run_code_ujb.sh eval chat codeujbrepair gpt-3.5-turbo gpt-3.5-turbo
./scripts/run_code_ujb.sh eval chat codeujbdefectdetection gpt-3.5-turbo gpt-3.5-turbo
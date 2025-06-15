import time
import yaml
from openai import OpenAI
from dotenv import load_dotenv
import os
from pprint import pprint
from IPython.display import display, Markdown
from pydantic import BaseModel, Field

class SimulationAgent:
    def __init__(self, api_key, output_limit=1000, Vector_store_id = None): 
        self.client = OpenAI(api_key=api_key)
        self.api_key = api_key
        self.output_limit = output_limit
        self.Vector_store_id = Vector_store_id

        if self.Vector_store_id == None:
            self.vector_store = None
        else:
            self.vector_store = self.retrieve_store()

        self.agents = []
        

    def run_completion(self, user_prompt):
            planner = self.agents[0]
            coder = self.agents[1]
            

            self.planner_thread = self.client.beta.threads.create()
            print("planner thread")
        
            self.client.beta.threads.messages.create(
                thread_id=self.planner_thread.id,
                role="user",
                content=user_prompt
            )
            print("messages created")

            self.planner_run = self.client.beta.threads.runs.create(
                thread_id=self.planner_thread.id,
                assistant_id=planner.id
            )
            print("planner run")
            i = 0
            while self.planner_run.status != "completed":
                time.sleep(1)
                print(i)
                self.planner_run = self.client.beta.threads.runs.retrieve(
                    thread_id=self.planner_thread.id,
                    run_id=self.planner_run.id
                )
                i += 1
        
            planner_output = None
            messages = self.client.beta.threads.messages.list(thread_id=self.planner_thread.id).data
            for msg in reversed(messages):
                if msg.role == "assistant":
                    planner_output = msg.content if isinstance(msg.content, str) else msg.content[0].text.value
                    break
                
            if not planner_output:
                raise RuntimeError("Planner returned no output.")
        
            print("\n--- Planner Output ---\n")
            print(planner_output)
        
            self.coder_thread = self.client.beta.threads.create()
        
            self.client.beta.threads.messages.create(
                thread_id=self.coder_thread.id,
                role="user",
                content=planner_output
            )
        
            self.coder_run = self.client.beta.threads.runs.create(
                thread_id=self.coder_thread.id,
                assistant_id=coder.id
            )
        
            while self.coder_run.status != "completed":
                time.sleep(1)
                self.coder_run = self.client.beta.threads.runs.retrieve(
                    thread_id=self.coder_thread.id,
                    run_id=self.coder_run.id
                )
        
            coder_output = None
            messages = self.client.beta.threads.messages.list(thread_id=self.coder_thread.id).data
            for msg in reversed(messages):
                if msg.role == "assistant":
                    coder_output = msg.content if isinstance(msg.content, str) else msg.content[0].text.value
                    break
                
            print("\n--- Coder Output ---\n")
            print(coder_output)
            return coder_output


    def create_agent(self):

        if self.vector_store is None:
            print("your vector store is not created yet")
            print("creating vector store now")
            self.vector_store = self.create_vector_store()

        with open("agents/planner_instruct.yaml", "r") as file:
            config = yaml.safe_load(file)
            instructions_planner = config.get("instructions")

        with open("agents/coder_instruct.yaml", "r") as file:
            config = yaml.safe_load(file)
            instructions_coder = config.get("instructions")

        print("creating planner agent...")
        planner = self.client.beta.assistants.create(
            name="Planner Agent",
            instructions=instructions_planner,
            tools=[{"type": "file_search", "file_search": {"max_num_results": 15}}],
            tool_resources={"file_search": {"vector_store_ids": [self.vector_store.id]}},
            model='gpt-4o-mini',
            temperature=0.0,
            top_p=1.0,
            response_format="auto"
        )
        self.agents.append(planner)

        print("creating coding agent...")
        coding = self.client.beta.assistants.create(
          name="Coding Agent",
          instructions=instructions_coder,
          tools=[
              {"type": "code_interpreter"},
              {"type": "file_search", "file_search": {"max_num_results": 5}}
          ],
          tool_resources={"file_search": {"vector_store_ids": [self.vector_store.id]}},
          model='gpt-4o-mini',
          temperature=0.0,
          top_p=1.0
      )
        self.agents.append(coding)

        print("Agents created")

    def create_vector_store(self):

        chunking_strategy = {
            "type": 'static',
            'static': {
                "max_chunk_size_tokens": 400,
                "chunk_overlap_tokens": 200
            }
        }
        start = time.time()
        assistant_data = self.load_directory()

        print("Files to upload:")
        file_paths = []
        for root, dirs, files in os.walk(assistant_data):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                print(" -", file)
                if file.startswith('.'):
                    continue
                file_paths.append(os.path.join(root, file))

        if not file_paths:
            print("No files found in data/ to upload.")
            return self.client.vector_stores.create(
                name="Cosmology_store",
                chunking_strategy=chunking_strategy
            )

        vector_store = self.client.vector_stores.create(
            name="Cosmology_store",
            chunking_strategy=chunking_strategy
        )

        streams = [open(path, "rb") for path in file_paths]
        try:
            self.client.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id,
                files=streams
            )
        finally:
            for s in streams:
                s.close()

        end = time.time()
        print(f"DEBUG: upload timing: {end - start:.2f} seconds")

        print("\n*********************************************************************************************************")
        print(f"THE ID FOR THE VECTOR STORE TO WHICH YOU ARE USING IS {vector_store.id} ***************************")
        print("*********************************************************************************************************\n")

        return vector_store


    def load_directory(self):
        os.makedirs("data", exist_ok=True)
        return os.path.abspath("data")

    def delete_agents(self):
        for a in self.agents:
            self.client.beta.assistants.delete(a.id)
        self.agents = []

    def delete_vect_store(self):
        vector_store_id = self.vs.id
        try:
            self.client.vector_stores.delete(vector_store_id)
            print(f"Vector store {vector_store_id} deleted successfully.")
        except Exception as e:
            print(f"Error deleting vector store {vector_store_id}: {e}")

    def inspect_run_steps(self):
        if not hasattr(self, "run") or not hasattr(self, "thread"):
            print("No run or thread available to inspect. Make sure you called run_completion(...) first.")
            return

        run_steps = self.client.beta.threads.runs.list(
            thread_id=self.thread.id,
            run_id=self.run.id
        )

        for i, step in enumerate(run_steps.data):
            print(f"\n--- Step {i} ---")
            tool_calls = getattr(step.step_details, "tool_calls", None)
            if not tool_calls:
                print("  No tool_calls in this step.")
                continue

            for call in tool_calls:
                if getattr(call, "file_search", None):
                    results = call.file_search.results
                    if not results:
                        print("  No file_search results in this step.")
                    else:
                        for r_idx, result in enumerate(results):
                            print(f"\n  Result {r_idx}:")
                            metadata = getattr(result, "metadata", {}) or {}
                            doc_name = metadata.get("filename", "Unknown Filename")
                            print(f"    Document Name: {doc_name}")
                            try:
                                snippet = result.content.get("text", {}).get("value", "[no text]")
                            except:
                                snippet = str(result.content)
                            print(f"    Snippet: {snippet[:200]}{'...' if len(snippet) > 200 else ''}")

    
    def retrieve_store(self):
        stores = self.client.vector_stores.list()
        for vs in stores:
            if vs.id == self.Vector_store_id:
                self.vector_store = vs
                return self.vector_store


    def view_steps(self):

        # write two seperate parts to this function, one for the planner thread, one for the coder thread

        # planner

        print("\n ****************************************************************************************************** ")
        print("********************************** STEPS TAKEN BY THE PLANNER AGENT **********************************")
        print("****************************************************************************************************** \n")

        run_steps = self.client.beta.threads.runs.steps.list(
            thread_id= self.planner_thread.id,
            run_id= self.planner_run.id
        )

        i = 0
        for step in run_steps.data:
            print("i: ", i)
            try:
            
                retrieved_step = self.client.beta.threads.runs.steps.retrieve(
                    thread_id=step.thread_id,
                    run_id= self.planner_run.id,
                    step_id=step.id,
                    include=["step_details.tool_calls[*].file_search.results[*].content"]
                )
                r = 0
                for result in retrieved_step.step_details.tool_calls[0].file_search.results:
                    print("\n\nr: ", r)
                    print("\n\nresult: ", result)
                    r += 1

            except:
                print("step.step_details.tool_calls: None")
            print("\n\nstep done\n\n")
            i += 1


        print(" \n ******************************************************************************************************")
        print("********************************** STEPS TAKEN BY THE CODER AGENT ************************************")
        print("****************************************************************************************************** \n")

        # coder

        run_steps = self.client.beta.threads.runs.steps.list(
            thread_id= self.coder_thread.id,
            run_id= self.coder_run.id
        )

        i = 0
        for step in run_steps.data:
            print("i: ", i)
            try:
            
                retrieved_step = self.client.beta.threads.runs.steps.retrieve(
                    thread_id=step.thread_id,
                    run_id= self.coder_run.id,
                    step_id=step.id,
                    include=["step_details.tool_calls[*].file_search.results[*].content"]
                )
                r = 0
                for result in retrieved_step.step_details.tool_calls[0].file_search.results:
                    print("\n\nr: ", r)
                    print("\n\nresult: ", result)
                    r += 1

            except:
                print("step.step_details.tool_calls: None")
            print("\n\nstep done\n\n")
            i += 1
            
        
        

    # Prof bolliet likely utilized the google API as a method of actually creating his Research agent!!! Try incorporating
    # most certaintly, you MUST add in a scrutiny agent, it must share the same vector store, and its main job is to scrutinize the plan.
        # This should run once, before returning to the planning agent, and then back to the coding agent

    # Add in the parallel processing code to which could make this run faster, it takes like a minute or 3 to upload files and run agents.

    # would it be better to create multiple vector stores named different things, and then add them to the agents.
        # Adds more organization, thus makes RAG search less expensive
            # - STAR Theory store
            # - STAR Cosmological Data store
            # - STAR Guide store.
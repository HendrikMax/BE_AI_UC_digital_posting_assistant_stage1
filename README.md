# BE_AI_UC_digital-posting_assistant_build1
SAP application digital posting assistant - build 1: 
Module A: vectorization accounting assignment guide in SAP HANA DB
Module B: account assignment support with RAG

## Program Description

### Module A: Vectorization of the Accounting Assignment Guide
Module A is used for processing and vectorizing an accounting assignment guide. The main functions include:
- **A0 - Preparation**: Setting up the environment, installing Python packages, and configuring connections to SAP HANA and SAP AI Hub.
- **A2 - Loading and Splitting**: Loading the PDF file of the accounting assignment guide and splitting it into text sections.
- **A3 - Vectorization**: Vectorizing the text sections using an embedding function.
- **A4 - Saving**: Saving the vectors in an SAP HANA database.

### Module B: Account Assignment Support
Module B supports the creation of account assignment information based on user input. The main functions include:
- **B0 - Preparation**: Setting up the environment, installing Python packages, and configuring connections to SAP HANA and SAP AI Hub.
- **B1 - Input**: Capturing the user's booking information via a user interface.
- **B2 - Retrieval**: Retrieving relevant account assignment information from the SAP HANA database.
- **B3 - Response**: Creating relevant account assignment information for the business case using an LLM.
- **B4 - Output**: Outputting the account assignment information to the user via the user interface.

## Usage Instructions

### Running the Application

1. **Install Requirements**:
   Ensure all required Python packages are installed. Run the following command in your terminal:
   ```bash
   pip install -r BE_AI_DPA_APP/requirements.txt
   ```

2. **Run the Application**:
   Navigate to the `BE_AI_DPA_APP` directory and execute the `app.py` file:
   ```bash
   python BE_AI_DPA_APP/app.py
   ```

   This will start the application. Follow the on-screen instructions to interact with the digital posting assistant.

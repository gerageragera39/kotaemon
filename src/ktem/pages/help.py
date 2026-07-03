import gradio as gr

class HelpPage:
    def __init__(self, app):
        self._app = app

        help_html = """
        <div class="legal-container">
          <div class="legal-header">
            <h1>Help & Legal Center</h1>

          </div>
          
          <div class="legal-layout">
            <!-- Left Column: Help & FAQs -->
            <div class="legal-column">
              <div class="legal-section-header">
                <h2>Help & FAQ</h2>
              </div>
              
              <div class="legal-item">
                <h3>What is the primary function of this assistant?</h3>
                <p>This platform is configured to assist students and academic advisors with queries related to the D3B Study program. It utilizes Retrieval-Augmented Generation (RAG) to query academic regulations, module descriptions, and schedules.</p>
              </div>
              
              <div class="legal-item">
                <h3>How do I query specific documents?</h3>
                <p>You can upload PDF or text files via the "File Collection" tab. In the Chat tab, expand the left sidebar and select the specific files you want to use as context for your questions. Choosing "Search All" will query all uploaded files in your database.</p>
              </div>
              
              <div class="legal-item">
                <h3>How should I verify the chatbot's answers?</h3>
                <p>You can check the "Information panel" on the right side of the chat screen. It displays the retrieved text passages, source document names, and relevance scores used to generate the response.</p>
              </div>
            </div>
            
            <!-- Right Column: Legal Notices -->
            <div class="legal-column">
              <div class="legal-section-header">
                <h2>Legal & Compliance</h2>
              </div>
              
              <div class="legal-item">
                <h3>1. Accuracy and Disclaimer</h3>
                <p>The information provided by this assistant is generated automatically. The university does not warrant the completeness or correctness of the replies. For legally binding academic decisions, students must refer to the official D3B Examination Regulations (Prüfungsordnung) or contact the student administration office.</p>
              </div>
              
              <div class="legal-item">
                <h3>2. Data Privacy & Local Processing</h3>
                <p>All input questions, chat history, and uploaded files are processed and stored exclusively on secure local servers managed by the university. No data is sent to external cloud providers or third-party AI companies.</p>
              </div>
              
              <div class="legal-item">
                <h3>3. Contact and IT Support</h3>
                <p>For administrative assistance, technical system issues, or data deletion requests, contact the IT support desk via the official email: <a href="mailto:support@ku.de">support@ku.de</a>.</p>
              </div>
            </div>
          </div>
        </div>
        """
        
        gr.HTML(help_html)

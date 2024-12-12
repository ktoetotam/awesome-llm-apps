import streamlit as st
from phi.agent import Agent
from phi.knowledge.text import TextKnowledgeBase
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.azure import AzureOpenAIChat
from phi.embedder.azure_openai import AzureOpenAIEmbedder
from phi.vectordb.pgvector import PgVector, SearchType
import logging
import tempfile
import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
logging.basicConfig(level=logging.DEBUG)

# Load environment variables from a .env file
load_dotenv()




def init_session_state():
    """
    Initialize session state variables if they don't exist.
    
    This function checks if certain keys are present in the Streamlit session state.
    If they are not, it initializes them with default values.
    """
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = None
    if 'azure_openai_endpoint' not in st.session_state:
        st.session_state.azure_openai_endpoint = None
    if 'search_client' not in st.session_state:
        st.session_state.search_client = None
    if 'ai_team' not in st.session_state:
        st.session_state.ai_team = None
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = None
    if 'ai_search' not in st.session_state:
        st.session_state.ai_search = None

def main():
    """
    Main function to run the Streamlit app for AI use case analysis.
    
    This function sets up the Streamlit app, initializes session state, loads API keys,
    configures the Azure OpenAI embedder, and sets up the AI agents and their roles.
    """
    # Set the page configuration for the Streamlit app
    st.set_page_config(page_title="AI Use Case Analyzer", layout="wide")
    init_session_state()

    # Title of the app
    st.title("AI Use Case Analyzer 🤖")
    
    # Load API keys and endpoints from environment variables
    openai_api_key = os.getenv('OPENAI_API_KEY')
    openai_api_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    search_api_key = os.getenv('AZURE_SEARCH_API_KEY')
    search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')

    # Initialize the Azure OpenAI embedder
    azure_openai_embedder = AzureOpenAIEmbedder(
        model="text-embedding-ada-002",   # The model you deployed on Azure
        azure_endpoint=openai_api_endpoint,
        api_key=openai_api_key,
        api_type="azure",
        api_version="2024-02-15-preview",
        deployment="text-embedding-ada-002"
    )

    # Get embeddings for the word "health"
    embeddings = azure_openai_embedder.get_embedding("health")
    print(embeddings)

    # Initialize the knowledge base with a vector database
    knowledge_base = TextKnowledgeBase(
        path="usecase.txt",
        vector_db=PgVector(
            table_name="data",
            search_type=SearchType.hybrid,
            db_url="postgresql+psycopg://ai:ai@localhost:5532/ai?gssencmode=disable",
            embedder=azure_openai_embedder
        ),
    )

    with st.sidebar:
        # Sidebar for API configuration
        st.header("🔑 API Configuration")
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key if st.session_state.openai_api_key else "",
            help="Enter your OpenAI API key"
        )
        st.session_state.openai_api_key = openai_api_key

        azure_openai_endpoint = st.text_input(
            "Azure OpenAI Endpoint",
            value=st.session_state.azure_openai_endpoint if st.session_state.azure_openai_endpoint else "",
            help="Enter your Azure OpenAI endpoint"
        )
        st.session_state.azure_openai_endpoint = openai_api_endpoint

        st.divider()
        # Load the knowledge base
        knowledge_base.load(recreate=True, upsert=True)
        print(knowledge_base)
        
        if st.session_state.openai_api_key:
            # Document upload section
            st.header("📄 Document Upload")
            uploaded_file = st.file_uploader("Upload AI Use Case Document", type=['txt', 'pdf'])
            
            # Initialize the Azure OpenAI chat model
            chat = AzureOpenAIChat(
                id="gpt-4o-mini",
                azure_endpoint=openai_api_endpoint,
                api_version="2024-02-15-preview",
                api_key=openai_api_key,
                azure_deployment="gpt-4o-mini"
            )

            # Define AI agents with specific roles and instructions
            ai_researcher = Agent(
                name="AI Researcher",
                role="AI research specialist",
                model=chat,
                tools=[DuckDuckGo()],
                knowledge_base=knowledge_base,
                instructions=[
                    "Find and cite relevant AI research papers and articles",
                    "Provide detailed research summaries with sources",
                    "Reference specific sections from the uploaded document",
                    "Always search the knowledge base for relevant information"
                ],
                show_tool_calls=True,
                markdown=True,
                debug_mode=True
            )

            # Define the Data Privacy and Security Expert agent
            data_privacy_security_expert = Agent(
                name="Data Privacy and Security Expert",
                role="Data privacy and security specialist",
                model=chat,
                knowledge_base=knowledge_base,
                instructions=[
                    "Evaluate the data privacy implications of the AI use case",
                    "Assess the security risks and recommend mitigation strategies",
                    "Ensure compliance with relevant data protection regulations"
                ],
                markdown=True,
                debug_mode=True
            )

            ai_analyst = Agent(
                name="AI Analyst",
                role="AI analysis specialist",
                model=chat,
                knowledge=knowledge_base,
                instructions=[
                    "Analyze the AI use case thoroughly",
                    "Identify key components and potential issues",
                    "Reference specific sections from the document",
                ],
                markdown=True,
                debug_mode=True
            )

            ai_strategist = Agent(
                name="AI Strategist", 
                role="AI strategy specialist",
                model=chat,
                knowledge_base=knowledge_base,
                instructions=[
                    "Develop comprehensive AI strategies",
                    "Provide actionable recommendations",
                    "Consider both risks and opportunities"
                ],
                markdown=True,
                debug_mode=True
            )

            # Define the Retrieval-Augmented Generation Expert agent
            rag_expert = Agent(
                name="Retrieval-Augmented Generation Expert",
                role="RAG specialist",
                model=chat,
                knowledge_base=knowledge_base,
                instructions=[
                    "Utilize retrieval-augmented generation techniques to enhance responses",
                    "Combine retrieved information with generated content",
                    "Ensure the accuracy and relevance of the generated responses"
                ],
                markdown=True
            )

            # Define the AI team lead agent
            st.session_state.ai_team = Agent(
                name="AI Team Lead",
                role="AI team coordinator",
                model=chat,
                team=[ai_analyst, ai_strategist, ai_researcher, data_privacy_security_expert,rag_expert],
                knowledge_base=knowledge_base,
                instructions=[
                    "Coordinate analysis between team members",
                    "Provide comprehensive responses",
                    "Ensure all recommendations are properly sourced",
                    "Reference specific parts of the uploaded document"
                ],
                show_tool_calls=True,
                markdown=True,
                debug_mode=True
            )

            st.divider()
            # Analysis options section
            st.header("🔍 Analysis Options")
            analysis_type = st.selectbox(
                "Select Analysis Type",
                [
                    "Use Case Review",
                    "AI Research",
                    "Risk Assessment",
                    "Feasibility Check",
                    "Data Privacy and Security",
                    "Retrieval-Augmented Generation",
                    "Custom Query"
                ]
            )
        else:
            st.warning("Please configure all API credentials to proceed")

    if not st.session_state.openai_api_key:
        st.info("👈 Please configure your API credentials in the sidebar to begin")
    elif not uploaded_file:
        st.info("👈 Please upload an AI use case document to begin analysis")
    elif st.session_state.ai_team:
        # Define icons for each analysis type
        analysis_icons = {
            "Use Case Review": "📑",
            "AI Research": "🔍",
            "Risk Assessment": "⚠️",
            "Feasibility Check": "✅",
            "Data Privacy and Security": "🔒",
            "Retrieval-Augmented Generation": "🔍",
            "Custom Query": "💭"
        }

        st.header(f"{analysis_icons[analysis_type]} {analysis_type} Analysis")
  
        # Define configurations for each analysis type
        analysis_configs = {
            "Use Case Review": {
                "query": "Review this AI use case and identify key components, benefits, and potential issues.",
                "agents": ["AI Analyst"],
                "description": "Detailed use case analysis focusing on components and benefits"
            },
            "AI Research": {
                "query": "Research relevant AI technologies and methodologies related to this use case.",
                "agents": ["AI Researcher"],
                "description": "Research on relevant AI technologies and methodologies"
            },
            "Risk Assessment": {
                "query": "Analyze potential risks and challenges in this AI use case.",
                "agents": ["AI Analyst", "AI Strategist"],
                "description": "Combined risk analysis and strategic assessment"
            },
            "Feasibility Check": {
                "query": "Check this AI use case for feasibility and implementation challenges.",
                "agents": ["AI Researcher", "AI Analyst", "AI Strategist"],
                "description": "Comprehensive feasibility analysis"
            },
            "Data Privacy and Security": {
                "query": "Evaluate the data privacy and security implications of this AI use case.",
                "agents": ["Data Privacy and Security Expert"],
                "description": "Analysis of data privacy and security aspects"
            },
            "Custom Query": {
                "query": None,
                "agents": ["AI Researcher", "AI Analyst", "AI Strategist", "Retrieval-Augmented Generation Expert", "Data Privacy and Security Expert"],
                "description": "Custom analysis using all available agents"
            },
            "Retrieval-Augmented Generation": {
                "query": "Check this AI use case and think about challenges from the point of view of retrieval-augmented generation techniques.",
                "agents": ["Retrieval-Augmented Generation Expert"],
                "description": "Comprehensive analysis of how RAG techniques can improve response accuracy and relevance as well as the analysis of limitations"
            }
        }

        st.info(f"📋 {analysis_configs[analysis_type]['description']}")
        st.write(f"🤖 Active AI Agents: {', '.join(analysis_configs[analysis_type]['agents'])}")

        if analysis_type == "Custom Query":
            user_query = st.text_area(
                "Enter your specific query:",
                help="Add any specific questions or points you want to analyze"
            )
        else:
            user_query = None

        if st.button("Analyze"):
            if analysis_type == "Custom Query" and not user_query:
                st.warning("Please enter a query")
            else:
                with st.spinner("Analyzing document..."):
                    if analysis_type != "Custom Query":
                        combined_query = f"""
                        Using the uploaded document as reference:
                        
                        Primary Analysis Task: {analysis_configs[analysis_type]['query']}
                        Focus Areas: {', '.join(analysis_configs[analysis_type]['agents'])}
                        
                        Please search the knowledge base and provide specific references from the document.
                        """
                    else:
                        combined_query = f"""
                        Using the uploaded document as reference:
                        
                        {user_query}
                        
                        Please search the knowledge base and provide specific references from the document.
                        Focus Areas: {', '.join(analysis_configs[analysis_type]['agents'])}
                        """

                    response = st.session_state.ai_team.run(combined_query)
                    
                    tabs = st.tabs(["Analysis", "Key Points", "Recommendations"])
                    
                    with tabs[0]:
                        st.markdown("### Detailed Analysis")
                        if response.content:
                            st.markdown(response.content)
                        else:
                            for message in response.messages:
                                if message.role == 'assistant' and message.content:
                                    st.markdown(message.content)
                    
                    with tabs[1]:
                        st.markdown("### Key Points")
                        key_points_response = st.session_state.ai_team.run(
                            f"""Based on this previous analysis:    
                            {response.content}
                            
                            Please summarize the key points in bullet points.
                            Focus on insights from: {', '.join(analysis_configs[analysis_type]['agents'])}"""
                        )
                        if key_points_response.content:
                            st.markdown(key_points_response.content)
                        else:
                            for message in key_points_response.messages:
                                if message.role == 'assistant' and message.content:
                                    st.markdown(message.content)
                    
                    with tabs[2]:
                        st.markdown("### Recommendations")
                        print("Recommendations")
                        recommendations_response = st.session_state.ai_team.run(
                            f"""Based on this previous analysis:
                            {response.content}
                            
                            Please summarize the recommendations in bullet points.
                            Focus on insights from: {', '.join(analysis_configs[analysis_type]['agents'])}"""
                        )
                        print("content found:",recommendations_response.content)
                        if recommendations_response.content:
                            st.markdown(recommendations_response.content)
                        else:
                            for message in recommendations_response.messages:
                                if message.role == 'assistant' and message.content:
                                    st.markdown(message.content)

if __name__ == "__main__":
    main()
import streamlit as st



from frontend.app import apply_custom_css

def render_beginner_guide():
    apply_custom_css()
    st.title("🎓 Cybersecurity 101: The Beginner's Guide")
    st.markdown("""
    Welcome to your personal cybersecurity classroom! 
    
    If the terminology on the dashboard looks like a foreign language, don't worry. This page explains everything you need to know using simple, real-world analogies.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("What is an IP Address?")
        st.info("💡 **Analogy:** Your home mailing address.")
        st.write("""
        An IP (Internet Protocol) address is simply the mailing address of your computer on the internet. 
        Just like the post office uses your home address to deliver mail, computers use IP addresses to send data (like a web page or an email) to each other.
        - `192.168.1.5` is an example of a local IP address (like an apartment number inside a building).
        """)

        st.subheader("What is a Port?")
        st.info("💡 **Analogy:** The doors and windows of your house.")
        st.write("""
        If your IP address is your house, a 'port' is a specific door or window. 
        - **Port 80** is the front door where web browsers (like Chrome or Safari) enter.
        - **Port 22** is the secure back door used by administrators.
        Hackers try to find 'open doors' to sneak into your computer.
        """)

        st.subheader("What is a Brute Force Attack?")
        st.info("💡 **Analogy:** A burglar trying every single key on your front door.")
        st.write("""
        A Brute Force attack is when a hacker uses a computer program to guess your password millions of times per second. 
        If you have a weak password like `password123`, they will get in very quickly. This is why our system watches for too many failed login attempts!
        """)

    with col2:
        st.subheader("What is a DDoS Attack?")
        st.info("💡 **Analogy:** A million people trying to enter a small store at the exact same time.")
        st.write("""
        DDoS stands for Distributed Denial of Service. It happens when hackers send massive amounts of junk traffic to a website or network, clogging it up so much that legitimate users (like you) cannot access it.
        Our system watches for huge spikes in traffic to detect this.
        """)

        st.subheader("What is a Port Scan?")
        st.info("💡 **Analogy:** A burglar walking around your house jiggling every door handle to see which ones are unlocked.")
        st.write("""
        Before hackers attack, they 'scan' your network to see which ports (doors) are open. 
        If our system detects someone jiggling all your digital door handles, it will raise an alert!
        """)

        st.subheader("What is SHAP (AI Explanation)?")
        st.info("💡 **Analogy:** A teacher asking a student to 'show their work' on a math test.")
        st.write("""
        Artificial Intelligence is often a 'black box'—it gives an answer, but doesn't explain *why*. 
        SHAP is a tool that forces the AI to show its math. It tells us exactly which pieces of data (like the length of the packet or the port used) convinced the AI that a hacker was attacking.
        """)

    st.markdown("---")
    st.success("🎉 You are now equipped with the basics! Return to the Live Monitor or Active Scanner to see these concepts in action.")

if __name__ == "__main__":
    render_beginner_guide()

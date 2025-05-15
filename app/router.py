from semantic_router import Route, SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.index.local import LocalIndex
from sklearn.metrics.pairwise import cosine_similarity


encoder = HuggingFaceEncoder(name="sentence-transformers/all-MiniLM-L6-v2")

faq = Route(
    name='faq',
    utterances=[
        # Existing ones
        "What is the refund policy of the products?",
        "Do I get discount with the HDFC credit card?",
        "How can I track my order?",
        "What payment methods are accepted?",
        "How long does it take to process a refund?",

        # Additional examples
        "Can I return a product after 10 days?",
        "What are the shipping charges?",
        "Is cash on delivery available?",
        "Do you offer EMI options?",
        "How can I cancel my order?",
        "Where is my order?",
        "Do you ship internationally?",
        "How do I apply a coupon code?",
        "Is my payment secure?",
        "Do you offer gift wrapping?",
        "Can I change my delivery address?",
        "What should I do if I received a damaged item?",
        "How do I contact customer support?",
        "How can I update my profile information?",
        "What is your exchange policy?",
        "How many days will it take to deliver my order?",
        "Are there any bank offers available right now?",
        "Can I track my return request?",
        "How do I know if my order is confirmed?",
        "Can I get an invoice for my order?",
        "What happens if I miss my delivery?",
        "Do you have a mobile app?",
        "Is there a loyalty program or rewards system?",
        "How do I delete my account?",
        "Can I get a GST invoice?",
        "What is your customer care number?"
    ]
)


sql = Route(
    name='sql',
    utterances=[
        # BRAND-based
        "Do you have Nike shoes?",
        "Show me products from Woodland",
        "List only Adidas sneakers",
        "I want Puma sandals with good reviews",

        # PRICE-based
        "Shoes under ₹3000",
        "I need sneakers below ₹2000",
        "Any deals under 2500 rupees?",
        "List premium shoes above ₹5000",

        # DISCOUNT-based
        "What are the biggest discounts right now?",
        "Any shoes on 50% discount?",
        "Show products with more than 30% off",
        "Best deals on running shoes?",

        # AVG_RATING-based
        "Top rated shoes",
        "Show me products with rating above 4.5",
        "I only want 5-star shoes",
        "List shoes with 4+ average rating",

        # TOTAL_RATINGS-based
        "Shoes with over 500 reviews",
        "Only show shoes rated by at least 1000 people",
        "Highly rated and popular products",
        "Products with many customer ratings",

        # PRODUCT TITLE-based
        "I want to see Campus women running shoes",
        "Show me Adidas Ultraboost models",
        "List all formal shoes",
        "Which shoes are best for gym?",

        # PRODUCT_LINK-based (implicit intent)
        "Show me all Puma sneakers with links",
        "Where can I buy Reebok shoes?",
        "Send me links to running shoes under ₹2000",
        "Give me the product pages for Nike shoes with 4+ ratings",

        # COMBINED FIELD EXAMPLES
        "Nike running shoes under ₹3000 with 4.5+ rating",
        "Adidas shoes with more than 1000 ratings and 30% off",
        "Formal shoes from Bata with rating above 4 and price below 2000",
        "Campus shoes with 50% discount and rating above 4.2",
        "Best Puma products under ₹4000 and minimum 500 ratings",
    ]
)

small_talk = Route(
    name='small-talk',
    utterances=[
        # Existing ones
        "How are you?",
        "What is your name?",
        "Are you a robot?",
        "What are you?",
        "What do you do?",

        # Additional examples
        "Hello!",
        "Hi there!",
        "Good morning!",
        "Good evening!",
        "What's up?",
        "How's your day going?",
        "Can we be friends?",
        "Do you sleep?",
        "Do you have feelings?",
        "Are you human?",
        "Do you know me?",
        "Tell me a joke.",
        "Can you make me laugh?",
        "Do you eat?",
        "Do you like music?",
        "Who made you?",
        "Where do you live?",
        "What can you do?",
        "Can you help me?",
        "Do you work 24/7?",
        "You're smart!",
        "You're funny.",
        "Nice to meet you!",
        "Thanks for your help!",
        "You're cool!",
        "Do you have a brain?",
        "Can you feel emotions?",
        "What's your purpose?"
    ]
)



routes = [faq, sql,small_talk]
utterances = []
expanded_routes = []

for route in routes:
    for utt in route.utterances:
        utterances.append(utt)
        expanded_routes.append(route.name)  # Use route name instead of the route object

embeddings = encoder(utterances)

index = LocalIndex()
index.add(embeddings=embeddings, utterances=utterances, routes=expanded_routes)

router = SemanticRouter(routes=routes, encoder=encoder, index=index)
# if __name__ == "__main__":
#     print(router("How can I track my order?").name)
#     print(router("Any shoes under Rs2000").name)

# query = "what is refund policy?"
# query_embedding = encoder([query])

# Compare query with all utterances
# similarities = cosine_similarity(query_embedding, embeddings)
# best_match_index = similarities.argmax()

# print(f"Query: {query}")
# print(f"Best Match: {utterances[best_match_index]}")
# print(f"Similarity score: {similarities[0][best_match_index]}")
# print(f"Matched Route: {expanded_routes[best_match_index]}")

# Query the router and print the result
# result = router(query)
# if result is None:
#     print("No match found!")
# else:
#     print(f"Matched Route: {result.name}")

import base64
import io
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from retrieval.search_engine import SearchEngine

app = FastAPI(title='Fashion Retrieval API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

engine = SearchEngine()


def get_image_data_url(image_id: int):
    try:
        image_idx = engine.image_ids.index(image_id)
        sample = engine.query_parser.parser.get_sample(image_idx)
    except Exception:
        return None

    image_obj = getattr(sample, 'image', None)
    if image_obj is None:
        return None

    try:
        if hasattr(image_obj, 'mode') and hasattr(image_obj, 'save'):
            img = image_obj.convert('RGB') if getattr(image_obj, 'mode', None) in {'RGBA', 'LA', 'P'} else image_obj
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
            return f'data:image/png;base64,{encoded}'

        if isinstance(image_obj, dict):
            image_array = image_obj.get('image') or image_obj.get('data')
            if image_array is None:
                return None
            img = Image.fromarray(image_array)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
            return f'data:image/png;base64,{encoded}'

        if hasattr(image_obj, '__array__'):
            img = Image.fromarray(image_obj)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
            return f'data:image/png;base64,{encoded}'
    except Exception:
        pass

    return None


@app.get('/health')
def health():
    return {'status': 'ok'}

@app.get('/search')
def search(q: str = Query(...)):
    parsed_query = engine.query_parser.parse(q)
    results = engine.retrieve(q)
    return {
        'query': q,
        'parsed_terms': sorted(parsed_query.get('extracted_terms', [])),
        'results': [
            {
                'image_id': item['image_id'],
                'final_score': item['final_score'],
                'embedding_score': item['embedding_score'],
                'category_score': item['category_score'],
                'color_score': item['color_score'],
                'style_score': item['style_score'],
                'environment_score': item['environment_score'],
                'object_score': item['object_score'],
                'num_objects': item['num_objects'],
                'categories': item.get('categories', []),
                'component_scores': item.get('component_scores', {}),
                'image_url': get_image_data_url(item['image_id']),
            }
            for item in results[:10]
        ],
    }

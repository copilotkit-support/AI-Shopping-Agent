"use client"

import { ProductCard } from "@/components/product-card"
import { Loader2, Package, Heart, FileText } from "lucide-react"
import { Button } from "@/components/ui/button"

interface Product {
  id: string
  image_urls: string[]
  title: string
  specifications: object
  review_sentiment: {
    positive_score: number,
    neutral_score: number,
    negative_score: number,
  },
  key_insights_from_reviews: string[],
  recommendation_score_out_of_100: number,
  would_buy_again_score_out_of_100: number,
  product_url: string,
  price_text: string,
  rating_value: number,
  rating_count: number,
  price: string
  image: string
  pros: string[]
  cons: string[]
  source: string
  rating: number
  reviews: number
  specs: {
    processor: string
    ram: string
    storage: string
    display: string
    battery: string
    weight: string
    ports: string
    os: string
  }
}

interface CanvasProps {
  products: Product[]
  isLoading: boolean
  query: string
  wishlist: string[]
  onToggleWishlist: (productId: string) => void
  onDeleteProduct: (productId: string) => void
  onGoToWishlist: () => void
  onGoToReport: () => void
  start: () => void
  report: any
  show_results: boolean
  wishlistLength: number
  canvasLogs: {
    title: string
    subtitle: string
  }
}

export function Canvas({
  products,
  canvasLogs,
  isLoading,
  query,  
  wishlist,
  wishlistLength,
  show_results,
  onToggleWishlist,
  onDeleteProduct,
  onGoToWishlist,
  onGoToReport,
  start,
  report,
}: CanvasProps) {
  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center bg-[#F7F7F9] relative">
        {/* Wishlist button - top right */}
        <Button
          onClick={onGoToWishlist}
          variant="outline"
          className="absolute top-6 right-6 border-[#D8D8E5] hover:bg-[#E8E8EF] bg-white flex items-center gap-2 z-10"
        >
          <Heart className="w-4 h-4" />
          Wishlist ({wishlistLength})
        </Button>
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-[#86ECE4] animate-spin mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-[#030507] mb-2 break-words">{(canvasLogs?.title == "" || canvasLogs?.title == undefined) ? "Processing your request..." : canvasLogs?.title} </h3>
          <p className="text-[#575758] break-words">{(canvasLogs?.subtitle == "" || canvasLogs?.subtitle == undefined) ? "" : canvasLogs?.subtitle}</p>
        </div>
        <div className="absolute bottom-6 inset-x-0 text-center">
          <p className="text-muted-foreground">May take around 5-10 minutes as it searches and compares items from different sites to find you the best options available</p>
        </div>
      </div>
    )
  }

  if (!query && !show_results) {
    return (
      <div className="flex-1 flex items-center justify-center bg-[#F7F7F9] relative">
        {/* Wishlist button - top right */}
        <Button
          onClick={onGoToWishlist}
          variant="outline"
          className="absolute top-6 right-6 border-[#D8D8E5] hover:bg-[#E8E8EF] bg-white flex items-center gap-2 z-10"
        >
          <Heart className="w-4 h-4" />
          Wishlist ({wishlistLength})
        </Button>
        <div className="text-center max-w-md">
          <Package className="w-16 h-16 text-[#858589] mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-[#030507] mb-2 font-['Roobert']">Ready to help you shop smarter</h3>
          <p className="text-[#575758] mb-6">
            Enter a search query to discover and compare products from across the web. I'll help you find the best deals
            and make informed decisions.
          </p>
          <div className="bg-gradient-to-r from-[#86ECCE] to-[#FFF388] p-6 rounded-xl">
            <h4 className="font-semibold text-[#030507] mb-2">Get Started</h4>
            <p className="text-sm text-[#030507]">Try searching for "wireless headphones" or "gaming laptop"</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 p-6 bg-[#F7F7F9] overflow-y-auto">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-2xl font-semibold text-[#030507] mb-2 font-['Roobert']">Product Recommendations</h2>
          <p className="text-[#575758]">
            Found {products.length} AI-curated products
          </p>
        </div>

        <div className="flex gap-3">
          <Button
            onClick={onGoToWishlist}
            variant="outline"
            className="border-[#D8D8E5] hover:bg-[#E8E8EF] bg-white flex items-center gap-2"
          >
            <Heart className="w-4 h-4" />
            Wishlist ({wishlistLength})
          </Button>
          <Button
            onClick={() => {
              debugger
              console.log(report, "report")
              if (!report) {
                start()
              }
              onGoToReport()
            }}
            disabled={ products.length === 0}
            // disabled={!query || products.length === 0}
            className="bg-[#1B606F] hover:bg-[#86ECE4] hover:text-[#030507] text-white flex items-center gap-2"
          >
            <FileText className="w-4 h-4" />
            View Report
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {products.map((product) => (
          <ProductCard
            key={product.id}
            product={product}
            isWishlisted={wishlist.map((product: any) => product.id).includes(product.id)}
            onToggleWishlist={onToggleWishlist}
            onDeleteProduct={onDeleteProduct}
          />
        ))}
      </div>
    </div>
  )
}

"use client"

import { ArrowLeft, Heart, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ProductCard } from "@/components/product-card"

interface Product {
  id: string
  name: string
  price: string
  image: string
  pros: string[]
  cons: string[]
  source: string
  image_urls: string[]
  title: string
  product_url: string
  price_text: string
  rating_value: number
  rating_count: number
  specifications: object
  review_sentiment: {
    positive_score: number,
    neutral_score: number,
    negative_score: number,
  },
  key_insights_from_reviews: string[],
  recommendation_score_out_of_100: number,
  would_buy_again_score_out_of_100: number,
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

interface WishlistViewProps {
  products: Product[]
  onExit: () => void
  onToggleWishlist: (productId: string) => void
  onDeleteProduct: (productId: string) => void
  clearAllWishlist: () => void
}

export function WishlistView({ products, onExit, onToggleWishlist, onDeleteProduct, clearAllWishlist }: WishlistViewProps) {


  if (products.length === 0) {
    return (
      <div className="flex-1 bg-[#F7F7F9]">
        {/* Header */}
        <div className="bg-white border-b border-[#D8D8E5] p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button variant="outline" onClick={onExit} className="border-[#D8D8E5] hover:bg-[#E8E8EF] bg-white">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Products
              </Button>
              <div>
                <h1 className="text-2xl font-semibold text-[#030507] font-['Roobert']">My Wishlist</h1>
                <p className="text-[#575758]">Your saved products</p>
              </div>
            </div>
          </div>
        </div>

        {/* Empty State */}
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center max-w-md">
            <Heart className="w-16 h-16 text-[#858589] mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-[#030507] mb-2 font-['Roobert']">Your wishlist is empty</h3>
            <p className="text-[#575758] mb-6">
              Start adding products to your wishlist by clicking the heart icon on any product card. This way you can
              save items you're interested in and compare them later.
            </p>
            <Button onClick={onExit} className="bg-[#030507] hover:bg-[#575758] text-white">
              Browse Products
            </Button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 bg-[#F7F7F9] overflow-y-auto">
      {/* Header */}
      <div className="bg-white border-b border-[#D8D8E5] p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button variant="outline" onClick={onExit} className="border-[#D8D8E5] hover:bg-[#E8E8EF] bg-white">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Products
            </Button>
            <div>
              <h1 className="text-2xl font-semibold text-[#030507] font-['Roobert']">My Wishlist</h1>
              <p className="text-[#575758]">{products.length} saved products</p>
            </div>
          </div>

          <div className="flex gap-3">
            <Button
              onClick={clearAllWishlist}
              variant="outline"
              className="border-[#FFA254] text-[#FFA254] hover:bg-[#FFA254] hover:text-white flex items-center gap-2 bg-transparent"
            >
              <Trash2 className="w-4 h-4" />
              Clear All
            </Button>
          </div>
        </div>
      </div>

      {/* Wishlist Content */}
      <div className="p-6">
        <div className="mb-6 hidden">
          <div className="bg-gradient-to-r from-[#86ECCE] to-[#FFF388] p-4 rounded-xl">
            <h3 className="font-semibold text-[#030507] mb-1">💡 Pro Tip</h3>
            <p className="text-sm text-[#030507]">
              Select multiple products using the compare button to get detailed side-by-side analysis and AI
              recommendations!
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {products.map((product) => (
            <ProductCard
              key={product.id}
              product={product}
              isWishlisted={true}
              onToggleWishlist={onToggleWishlist}
              onDeleteProduct={onDeleteProduct}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

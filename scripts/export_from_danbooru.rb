#!/usr/bin/env ruby

# dump data for inception trainer
n = 500_000
max_id = Post.maximum(:id)
CurrentUser.user = User.admins.first
CurrentUser.user.disable_tagged_filenames = true
CSV.open("/tmp/posts_chars.csv", "w") do |csv|
  csv << ["id", "md5", "url", "character"]
  Post.without_timeout do
    Post.where("id > ? and tag_count_character = 1", max_id - n).find_each do |post|
      if post.file_ext == "jpg" && !post.has_tag?("comic")
      	url = post.large_file_url.sub(/\/data\//, "")
        csv << [post.id, post.md5, "https://s3.amazonaws.com/danbooru/#{url}", post.character_tags.join("")]
      end
    end
  end
end
